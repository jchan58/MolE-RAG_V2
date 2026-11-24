"""
Two-group passage chunker over downloaded JATS full-text XML (v1).

One PARAGRAPH per passage from <body>; each TABLE is a separate structured passage
(caption + column headers + row headers + units + footnotes; large tables split with
parent_table_id). Each article parsed ONCE; both group tags/associations if in both.

Refinements implemented:
 1. passage_id = hash(pmcid + source_location + chunker_version); content_hash =
    hash(normalized text) is a SEPARATE field (identical text -> distinct passage_ids;
    never collapse by content).
 2. Offsets anchored to a saved canonical representation: extracted_text/<shard>/
    PMCID.v<ver>.txt; each passage stores source_repr, source_repr_version,
    source_repr_hash and char_start/char_end into THAT text.
 3. collected_for (article-level collection provenance: groups + molecule_ids) is
    DISTINCT from any passage-level judge identity.
 4. Tables keep full context (caption, column + row headers, units, footnotes) and
    link split parts via parent_table_id.
 5. Missing metadata explicit: null / [] + per-field metadata_status + metadata_source;
    nothing invented.

Per-article outputs: passages/<shard>/<pmcid>.jsonl, extracted_text/<shard>/<pmcid>.v1.txt
Resumable: skip if the passage file already exists.
"""
import os, sys, json, hashlib, re
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import download as DL
import metadata as MD

CHUNKER_VERSION = "v1"
MEGA_LIMIT = 50000
ALIASES_PER_MOL = 24
TABLE_SPLIT_CHARS = 8000


def _ln(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(elem):
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip() if elem is not None else ""


def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def passages_path(pmcid):
    return C.BP("passages", M.shard_of(pmcid), "%s.jsonl" % pmcid)


def extracted_text_path(pmcid):
    return C.BP("extracted_text", M.shard_of(pmcid), "%s.%s.txt" % (pmcid, CHUNKER_VERSION))


def is_chunked(pmcid):
    return os.path.exists(passages_path(pmcid))


# ---------------- group membership + aliases ----------------
def load_groups():
    prop = set(str(p) for p in (C.read_json(C.P("data", "property_task_pmcids.json"), {}) or {}).get("pmcids", []))
    mol_map = C.read_json(C.P("data", "molecule_pmcids.json"), {}) or {}
    pmcid_mols = {}
    for mid, info in mol_map.items():
        if info.get("n_pmcids", 0) > MEGA_LIMIT:
            continue
        for p in info.get("pmcids", []):
            pmcid_mols.setdefault(str(p), set()).add(mid)
    return prop, pmcid_mols


def load_aliases():
    out = {}
    path = C.P("data", "molecule_identities_full.jsonl")
    src = path if os.path.exists(path) else C.P("data", "molecule_identities.jsonl")
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            syns = r.get("synonyms_full") or r.get("synonyms") or []
            seen, chosen = set(), []
            for s in sorted(syns, key=lambda x: (0 if any(c.isalpha() for c in x) else 1, len(x))):
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k); chosen.append(s)
                if len(chosen) >= ALIASES_PER_MOL:
                    break
            out[r["mol_id"]] = dict(
                preferred_name=r.get("preferred_name") or r.get("iupac_name"),
                iupac_name=r.get("iupac_name"), cid=r.get("cid"), aliases=chosen)
    return out


# ---------------- article metadata (explicit status/source) ----------------
def _article_meta(pmcid, art):
    md = C.read_json(MD._meta_path(pmcid))
    fields = {}
    def setf(name, esum_key, xml_getter):
        val, src, status = None, None, "missing"
        if md and md.get(esum_key) not in (None, "", []):
            val, src, status = md.get(esum_key), "esummary", "present"
        else:
            xv = xml_getter() if xml_getter else None
            if xv not in (None, "", []):
                val, src, status = xv, "jats_xml", "present"
        fields[name] = dict(value=val, metadata_status=status, metadata_source=src)
    def xtitle():
        t = art.find(".//front//article-title") if art is not None else None
        return _text(t) if t is not None else None
    def xjournal():
        j = art.find(".//front//journal-title") if art is not None else None
        return _text(j) if j is not None else None
    setf("article_title", "esummary_title", xtitle)
    setf("journal", "source_journal", xjournal)
    setf("pubdate", "pubdate", None)
    # authors (list)
    if md and md.get("authors"):
        fields["authors"] = dict(value=md.get("authors"), metadata_status="present", metadata_source="esummary")
    else:
        auth = []
        if art is not None:
            for nm in art.findall(".//front//contrib//name"):
                s = _text(nm)
                if s:
                    auth.append(s)
        fields["authors"] = dict(value=auth, metadata_status=("present" if auth else "missing"),
                                 metadata_source=("jats_xml" if auth else None))
    fields["doi"] = dict(value=(md.get("doi") if md else None),
                         metadata_status=("present" if (md and md.get("doi")) else "missing"),
                         metadata_source=("esummary" if (md and md.get("doi")) else None))
    return fields


# ---------------- tables ----------------
def _table_struct(tw):
    label = _text(tw.find(".//label")) if tw.find(".//label") is not None else None
    caption = _text(tw.find(".//caption")) if tw.find(".//caption") is not None else None
    col_headers = []
    for tbl in tw.findall(".//table"):
        thead = tbl.find(".//thead")
        if thead is not None:
            for tr in thead.iter():
                if _ln(tr.tag) == "tr":
                    col_headers.append([_text(c) for c in tr if _ln(c.tag) in ("th", "td")])
    rows = []
    for tbl in tw.findall(".//table"):
        body = tbl.find(".//tbody")
        scope = body if body is not None else tbl
        for tr in scope.iter():
            if _ln(tr.tag) == "tr":
                cells = [_text(c) for c in tr if _ln(c.tag) in ("th", "td")]
                if any(cells):
                    rows.append(cells)
    foot = _text(tw.find(".//table-wrap-foot")) if tw.find(".//table-wrap-foot") is not None else None
    return dict(label=label, caption=caption, column_headers=col_headers, rows=rows, footnotes=foot)


def _table_text(struct, rows=None):
    parts = []
    if struct["label"]:
        parts.append("LABEL: " + struct["label"])
    if struct["caption"]:
        parts.append("CAPTION: " + struct["caption"])
    for ch in struct["column_headers"]:
        parts.append("COLHDR: " + " | ".join(ch))
    for r in (rows if rows is not None else struct["rows"]):
        parts.append(" | ".join(r))
    if struct["footnotes"]:
        parts.append("FOOTNOTES: " + struct["footnotes"])
    return "\n".join(p for p in parts if p.strip())


# ---------------- walk ----------------
def _walk(elem, section_path, items, state):
    for child in elem:
        tag = _ln(child.tag)
        if tag == "sec":
            tnode = None
            for c in child:
                if _ln(c.tag) == "title":
                    tnode = c; break
            _walk(child, section_path + [_text(tnode)], items, state)
        elif tag == "p":
            txt = _text(child)
            if txt:
                items.append(("paragraph", txt, list(section_path), None, None))
        elif tag == "table-wrap":
            st = _table_struct(child)
            base_idx = state["tbl"]; state["tbl"] += 1
            full = _table_text(st)
            if len(full) <= TABLE_SPLIT_CHARS or not st["rows"]:
                items.append(("table", full, list(section_path), "T%d" % base_idx, st))
            else:
                # split rows into chunks, each carrying caption+headers+footnotes
                parent = "T%d" % base_idx
                chunk, cur = [], 0
                for r in st["rows"]:
                    chunk.append(r); cur += sum(len(x) for x in r)
                    if cur >= TABLE_SPLIT_CHARS:
                        items.append(("table", _table_text(st, chunk), list(section_path), parent, st))
                        chunk, cur = [], 0
                if chunk:
                    items.append(("table", _table_text(st, chunk), list(section_path), parent, st))
        elif tag in ("fig", "disp-formula", "label", "caption", "title", "table-wrap-foot"):
            continue
        else:
            _walk(child, section_path, items, state)


def chunk_article(pmcid, prop_set, pmcid_mols, aliases):
    xp = DL.xml_path(pmcid)
    if not os.path.exists(xp):
        return None
    try:
        root = ET.parse(xp).getroot()
    except ET.ParseError:
        C.atomic_write_text(passages_path(pmcid), "", fsync=False)  # mark done (empty)
        return dict(pmcid=pmcid, parse_error=True, n_paragraphs=0, n_tables=0,
                    in_property=str(pmcid) in prop_set, in_molecule=bool(pmcid_mols.get(str(pmcid))))
    art = root if _ln(root.tag) == "article" else root.find(".//article")
    body = art.find(".//body") if art is not None else None
    meta = _article_meta(pmcid, art)

    items = []
    state = dict(tbl=0)
    if body is not None:
        _walk(body, [], items, state)

    # canonical extracted text + offsets
    canon_parts = []
    offsets = []
    pos = 0
    for (ptype, txt, secpath, tid, st) in items:
        offsets.append((pos, pos + len(txt)))
        canon_parts.append(txt)
        pos += len(txt) + 2  # "\n\n" separator
    canon = "\n\n".join(canon_parts)
    etp = extracted_text_path(pmcid)
    C.atomic_write_text(etp, canon, fsync=False)
    repr_hash = hashlib.sha1(canon.encode("utf-8")).hexdigest()
    source_repr = os.path.basename(etp)

    in_prop = str(pmcid) in prop_set
    mol_ids = sorted(pmcid_mols.get(str(pmcid), set()))
    in_mol = bool(mol_ids)
    groups = (["property"] if in_prop else []) + (["molecule"] if in_mol else [])
    mol_assoc = [dict(molecule_id=m, preferred_name=(aliases.get(m, {}) or {}).get("preferred_name"),
                      iupac_name=(aliases.get(m, {}) or {}).get("iupac_name"),
                      cid=(aliases.get(m, {}) or {}).get("cid"),
                      aliases=(aliases.get(m, {}) or {}).get("aliases", [])) for m in mol_ids]
    collected_for = dict(groups=groups, molecule_ids=mol_ids,
                         note="ARTICLE-level collection provenance; distinct from per-passage judge identity_match")

    out = []
    n_par = n_tab = 0
    for i, (ptype, txt, secpath, tid, st) in enumerate(items):
        cs, ce = offsets[i]
        source_location = "%s/%s[%d]" % ("::".join(secpath), ptype, i)
        pid = hashlib.sha1(("%s|%s|%s" % (pmcid, source_location, CHUNKER_VERSION)).encode()).hexdigest()[:20]
        content_hash = hashlib.sha1(_norm(txt).encode("utf-8")).hexdigest()
        rec = dict(
            passage_id="PX_" + pid, pmcid=pmcid, pmcid_full="PMC%s" % pmcid,
            chunker_version=CHUNKER_VERSION,
            source_location=source_location,
            source_repr=source_repr, source_repr_version=CHUNKER_VERSION, source_repr_hash=repr_hash,
            char_start=cs, char_end=ce,
            content_hash=content_hash,
            passage_type=ptype, section_path=secpath, heading=(secpath[-1] if secpath else None),
            local_index=i,
            # explicit metadata with status/source
            article_title=meta["article_title"], journal=meta["journal"],
            pubdate=meta["pubdate"], authors=meta["authors"], doi=meta["doi"],
            metadata_source=("esummary" if C.read_json(MD._meta_path(pmcid)) else "jats_xml_or_missing"),
            document_groups=groups,
            collected_for=collected_for,
            molecule_associations=(mol_assoc if in_mol else []),
            text=txt,
        )
        if ptype == "table":
            n_tab += 1
            rec["parent_table_id"] = (("PX_tableparent_%s_%s" % (pmcid, tid)) if tid else None)
            rec["table"] = dict(label=st["label"], caption=st["caption"],
                                column_headers=st["column_headers"], footnotes=st["footnotes"],
                                n_rows=len(st["rows"]))
        else:
            n_par += 1
        out.append(rec)

    C.atomic_write_text(passages_path(pmcid),
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + ("\n" if out else ""),
                        fsync=False)
    return dict(pmcid=pmcid, n_paragraphs=n_par, n_tables=n_tab,
                in_property=in_prop, in_molecule=in_mol, n_molecules=len(mol_ids),
                parse_error=False)
