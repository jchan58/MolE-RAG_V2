"""
Stage 2 — PubChem identity resolution for each UNIQUE valid structure.

For each structure we attempt, in order: exact SMILES match, then InChIKey match.
We record CID(s), preferred name, IUPAC name, full synonyms, heuristic dev codes,
PubChem's returned SMILES/InChI/InChIKey (keeping PubChem's own property names),
the match type, a structure-verification outcome (RDKit canonical comparison), and
the raw response paths. Salts/stereoisomers are NOT merged (PubChem keeps them as
distinct CIDs; we verify the CID's structure canonicalizes to ours). Unresolved
and ambiguous structures are kept in the manifest and processing continues.

Per-item result: results/pubchem/<mol_id>.json  (idempotent; skipped if present)
Raw responses:   raw_pubchem/<mol_id>.<what>.json
"""
import os, sys, json, re, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

PROP_LIST = "Title,IUPACName,SMILES,ConnectivitySMILES,InChI,InChIKey,MolecularFormula,MolecularWeight"
DEV_CODE_RE = re.compile(r"^[A-Za-z]{1,6}[-\s]?\d{2,6}([-\s]?[A-Za-z0-9]{1,4})?$")


def _save_raw(mol_id, what, obj_or_text):
    path = C.P("raw_pubchem", "%s.%s.json" % (mol_id, what))
    if isinstance(obj_or_text, (dict, list)):
        C.write_json(path, obj_or_text)
    else:
        C.atomic_write_text(path, obj_or_text if obj_or_text is not None else "")
    return path


def _pubchem_json(path, mol_id, what):
    """GET a PubChem REST path returning JSON. Returns (status, json_or_None, raw_path, err)."""
    r = C.pubchem_get(path, want="text")
    raw_path = _save_raw(mol_id, what, r.text or (r.error or ""))
    if r.ok:
        try:
            return (200, json.loads(r.text), raw_path, None)
        except json.JSONDecodeError as e:
            return (200, None, raw_path, "json-decode: %r" % e)
    return (r.status, None, raw_path, r.error)


def _pubchem_post_cids_by_smiles(smiles, mol_id):
    url = C.ENDPOINTS["pubchem_base"] + "/compound/smiles/cids/JSON"
    r = C.http_request("pubchem", "POST", url, data={"smiles": smiles}, want="text")
    raw_path = _save_raw(mol_id, "smiles_cids", r.text or (r.error or ""))
    if r.ok:
        try:
            j = json.loads(r.text)
            return (200, j.get("IdentifierList", {}).get("CID", []), raw_path, None)
        except json.JSONDecodeError as e:
            return (200, [], raw_path, "json-decode: %r" % e)
    if r.status == 404:
        return (404, [], raw_path, "not-found")
    return (r.status, [], raw_path, r.error)


def _cids_by_inchikey(inchikey, mol_id):
    path = "/compound/inchikey/%s/cids/JSON" % urllib.parse.quote(inchikey, safe="")
    status, j, raw, err = _pubchem_json(path, mol_id, "inchikey_cids")
    if status == 200 and j:
        return (200, j.get("IdentifierList", {}).get("CID", []), raw, None)
    if status == 404:
        return (404, [], raw, "not-found")
    return (status, [], raw, err)


def _fetch_properties(cid, mol_id):
    path = "/compound/cid/%d/property/%s/JSON" % (cid, PROP_LIST)
    status, j, raw, err = _pubchem_json(path, mol_id, "properties")
    if status == 200 and j:
        props = j.get("PropertyTable", {}).get("Properties", [])
        return (props[0] if props else {}, raw, err)
    return ({}, raw, err)


def _fetch_synonyms(cid, mol_id):
    path = "/compound/cid/%d/synonyms/JSON" % cid
    status, j, raw, err = _pubchem_json(path, mol_id, "synonyms")
    if status == 200 and j:
        info = j.get("InformationList", {}).get("Information", [])
        if info:
            return (info[0].get("Synonym", []) or [], raw, err)
    return ([], raw, err)


def _verify_cid_structure(cid_props, our_canonical):
    """Compare the CID's isomeric SMILES (RDKit-canonicalized) to our structure."""
    cand = cid_props.get("SMILES") or cid_props.get("IsomericSMILES") \
        or cid_props.get("ConnectivitySMILES") or cid_props.get("CanonicalSMILES")
    if not cand:
        return dict(method="rdkit_canonical_match", outcome="no_pubchem_smiles",
                    pubchem_smiles=None, matches=False)
    m = Chem.MolFromSmiles(cand)
    if m is None:
        return dict(method="rdkit_canonical_match", outcome="pubchem_smiles_unparseable",
                    pubchem_smiles=cand, matches=False)
    can = Chem.MolToSmiles(m, isomericSmiles=True)
    exact = (can == our_canonical)
    # Fallback comparison on connectivity (stereo-insensitive) for reporting only.
    m2 = Chem.MolFromSmiles(our_canonical)
    conn_match = False
    if m2 is not None:
        conn_match = (Chem.MolToSmiles(m, isomericSmiles=False)
                      == Chem.MolToSmiles(m2, isomericSmiles=False))
    return dict(method="rdkit_canonical_match",
                outcome=("exact" if exact else
                         ("connectivity_only" if conn_match else "mismatch")),
                pubchem_smiles=cand, pubchem_canonical=can, matches=exact)


def _dev_codes(synonyms):
    out = []
    for s in synonyms:
        s = s.strip()
        if DEV_CODE_RE.match(s) and any(ch.isdigit() for ch in s) and len(s) <= 20:
            out.append(s)
    # de-dup preserving order
    seen, res = set(), []
    for s in out:
        if s.lower() not in seen:
            seen.add(s.lower()); res.append(s)
    return res


def resolve_structure(srec):
    """Resolve one structure record (from structures.jsonl) to a PubChem identity.
    Returns the identity dict. Idempotent-safe (pure given network)."""
    mol_id = srec["mol_id"]
    canonical = srec["canonical_smiles"]
    inchikey = srec.get("inchikey")

    identity = dict(
        mol_id=mol_id, canonical_smiles=canonical, inchikey_rdkit=inchikey,
        split=srec.get("split"), splits=srec.get("splits"),
        row_ids=srec.get("row_ids"),
        resolved_at=C.now_iso(),
        match_type=None, cid=None, candidate_cids=[],
        preferred_name=None, iupac_name=None, synonyms=[], dev_codes=[],
        pubchem_properties={}, verification=None,
        raw_response_paths={}, errors=[],
    )

    # 1) exact SMILES
    status, cids, raw, err = _pubchem_post_cids_by_smiles(canonical, mol_id)
    identity["raw_response_paths"]["smiles_cids"] = raw
    match_source = None
    if status == 200 and cids:
        match_source = "smiles"
    elif status in (404,) or (status == 200 and not cids):
        # 2) InChIKey fallback
        if inchikey:
            status2, cids2, raw2, err2 = _cids_by_inchikey(inchikey, mol_id)
            identity["raw_response_paths"]["inchikey_cids"] = raw2
            if status2 == 200 and cids2:
                cids = cids2; match_source = "inchikey"
            elif status2 not in (200, 404):
                identity["errors"].append("inchikey_cids: %s %s" % (status2, err2))
        if match_source is None:
            # genuine no-match vs failure
            if status == 404 or (status == 200):
                identity["match_type"] = "unresolved_no_match"
            else:
                identity["match_type"] = "unresolved_failure"
                identity["errors"].append("smiles_cids: %s %s" % (status, err))
            return identity
    else:
        identity["match_type"] = "unresolved_failure"
        identity["errors"].append("smiles_cids: %s %s" % (status, err))
        return identity

    identity["candidate_cids"] = cids
    # Resolve a single identity: verify each candidate, prefer exact-structure match.
    chosen = None
    verifications = {}
    for cid in cids[:10]:
        props, praw, perr = _fetch_properties(cid, "%s.cid%d" % (mol_id, cid))
        v = _verify_cid_structure(props, canonical)
        verifications[cid] = dict(verification=v, properties=props)
        if perr:
            identity["errors"].append("properties cid %d: %s" % (cid, perr))
        if v["outcome"] == "exact" and chosen is None:
            chosen = cid

    if chosen is None:
        if len(cids) == 1:
            chosen = cids[0]  # single candidate, accept with recorded verification
        else:
            # multiple candidates, none verified exact -> ambiguous
            identity["match_type"] = "ambiguous"
            identity["verification"] = {str(k): verifications[k]["verification"]
                                        for k in verifications}
            return identity

    identity["cid"] = chosen
    chosen_props = verifications.get(chosen, {}).get("properties", {})
    identity["pubchem_properties"] = chosen_props
    identity["verification"] = verifications.get(chosen, {}).get("verification")
    identity["iupac_name"] = chosen_props.get("IUPACName")
    identity["preferred_name"] = chosen_props.get("Title") or chosen_props.get("IUPACName")

    syns, sraw, serr = _fetch_synonyms(chosen, mol_id)
    identity["raw_response_paths"]["synonyms"] = sraw
    identity["synonyms"] = syns
    identity["dev_codes"] = _dev_codes(syns)
    if serr:
        identity["errors"].append("synonyms: %s" % serr)
    if not identity["preferred_name"] and syns:
        identity["preferred_name"] = syns[0]

    # match_type classification
    v = identity["verification"] or {}
    if match_source == "smiles" and v.get("outcome") == "exact":
        identity["match_type"] = "exact"
    elif v.get("outcome") == "exact":
        identity["match_type"] = "exact_via_inchikey"
    elif v.get("outcome") == "connectivity_only":
        identity["match_type"] = "normalized_connectivity"
    elif len(cids) > 1:
        identity["match_type"] = "ambiguous_resolved_single"
    else:
        identity["match_type"] = "normalized"
    return identity


def item_ids():
    ids = []
    with open(C.P("data", "structures.jsonl")) as f:
        for line in f:
            ids.append(json.loads(line)["mol_id"])
    return ids


def structures_by_id():
    d = {}
    with open(C.P("data", "structures.jsonl")) as f:
        for line in f:
            s = json.loads(line)
            d[s["mol_id"]] = s
    return d


def result_path(mol_id):
    return C.P("results", "pubchem", "%s.json" % mol_id)


def is_done(mol_id):
    return C.read_json(result_path(mol_id)) is not None


def process_item(mol_id, structs):
    if is_done(mol_id):
        return "cached"
    srec = structs[mol_id]
    identity = resolve_structure(srec)
    C.write_json(result_path(mol_id), identity)
    C.record_usage("pubchem_resolve", mol_id=mol_id, match_type=identity["match_type"])
    return identity["match_type"]
