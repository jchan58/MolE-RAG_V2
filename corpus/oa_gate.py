"""
OA-gating stage — determine PMC Open-Access full-text availability via the S3
Cloud Service bucket (presence of a PMC<id>. prefix == OA full text available).
This replaces "open access"[filter] gating and is OFF the E-utilities cap.

Items: registry PMCIDs. Per item we do ONE S3 LIST (cheap, parallel) and record
whether the article is in the bucket + the JATS .xml key + version.

Per-item result: results/oa_gate/<shard>/<pmcid>.json
Collated: data/oa_membership.jsonl, data/download_targets.txt, data/oa_summary.json
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import merge as M
import s3util as S3


def _res_path(pmcid):
    return C.P("results", "oa_gate", M.shard_of(pmcid), "%s.json" % pmcid)


def item_ids():
    return M.registry_pmcids()


def is_done(pmcid):
    return C.read_json(_res_path(pmcid)) is not None


def process_item(pmcid):
    if is_done(pmcid):
        return "cached"
    keys, err = S3.list_prefix_keys(pmcid)
    if err:
        # transient S3 error -> do not finalize; let lease/retry handle it
        raise RuntimeError("oa_gate list failed for %s: %s" % (pmcid, err))
    xml_key = S3.pick_xml_key(keys)
    rec = dict(pmcid=pmcid, pmcid_full="PMC%s" % pmcid,
               in_s3=bool(xml_key), xml_key=xml_key, n_keys=len(keys),
               checked_at=C.now_iso())
    C.write_json(_res_path(pmcid), rec, fsync=False)
    return "in_s3" if xml_key else "not_in_s3"


def collate():
    out = C.P("data", "oa_membership.jsonl")
    targets = []
    n_in = n_out = 0
    with open(out, "w") as f:
        for path in glob.glob(C.P("results", "oa_gate", "*", "*.json")):
            r = C.read_json(path)
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r.get("in_s3"):
                n_in += 1
                targets.append(r["pmcid"])
            else:
                n_out += 1
    targets.sort(key=lambda x: int(x) if str(x).isdigit() else x)
    C.atomic_write_text(C.P("data", "download_targets.txt"),
                        "\n".join(str(t) for t in targets) + ("\n" if targets else ""))
    summary = dict(built_at=C.now_iso(), n_registry=n_in + n_out,
                   n_oa_in_s3=n_in, n_non_oa=n_out)
    C.write_json(C.P("data", "oa_summary.json"), summary)
    C.log("INFO", "oa_gate", "OA membership: %d in S3 / %d non-OA (of %d registry)"
          % (n_in, n_out, n_in + n_out))
    return summary


def download_targets():
    path = C.P("data", "download_targets.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def xml_key_for(pmcid):
    r = C.read_json(_res_path(pmcid))
    return r.get("xml_key") if r else None
