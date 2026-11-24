"""
Parallel chunker driver — parses every downloaded JATS article across many CPU
cores into passages (chunker.py). Incremental/resumable (skips already-chunked
PMCIDs). Reports exact passage counts to state/CHUNK_COUNTS.json.
"""
import os, sys, json, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import chunker as CH

# Gentler default (16) to avoid Lustre MDS thrash from many concurrent small-file
# writers; override with EXP70_CHUNK_PROCS.
PROCS = int(os.environ.get("EXP70_CHUNK_PROCS", "16"))

# Built ONCE in the parent before the Pool forks; workers inherit via copy-on-write
# (the non-mega molecule inverse map is ~2.9M entries — must not be duplicated or
# rebuilt per worker).
_G = {}

def _work(pmcid):
    try:
        if CH.is_chunked(pmcid):
            return ("cached", pmcid, 0, 0, False, False)
        r = CH.chunk_article(pmcid, _G["prop"], _G["mols"], _G["aliases"])
        if r is None:
            return ("missing", pmcid, 0, 0, False, False)
        return (("err" if r.get("parse_error") else "ok"), pmcid,
                r["n_paragraphs"], r["n_tables"], r["in_property"], r["in_molecule"])
    except Exception:
        return ("err", pmcid, 0, 0, False, False)

def _clean_set():
    p = C.P("data", "clean_pmcids.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return set(x.strip() for x in f if x.strip())

def enumerate_downloaded():
    base = C.BP("fulltext_xml")
    if not os.path.isdir(base):
        return
    clean = _clean_set()  # restrict full-text processing to the clean set when present
    for shard in os.listdir(base):
        d = os.path.join(base, shard)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".xml"):
                    pmcid = fn[:-4]
                    if clean is None or pmcid in clean:
                        yield pmcid

def _counts0():
    return dict(articles_parsed=0, errors=0, cached=0, missing=0,
                paragraph_passages=0, table_passages=0,
                property_articles=0, molecule_articles=0, both_articles=0,
                property_passages=0, molecule_passages=0, both_passages=0)

def _write(counts, total, done, t0, final=False):
    rate = done / max(time.time() - t0, 1e-6)
    counts = dict(counts, updated_at=C.now_iso(), articles_todo=total, articles_done=done,
                  chunk_rate_per_s=round(rate, 1),
                  total_passages=counts["paragraph_passages"] + counts["table_passages"],
                  phase=("done" if final else "chunking"), chunker_version=CH.CHUNKER_VERSION)
    C.write_json(C.P("state", "CHUNK_COUNTS.json"), counts)

def run():
    C.ensure_dirs()
    os.makedirs(C.BP("passages"), exist_ok=True)
    os.makedirs(C.BP("extracted_text"), exist_ok=True)
    C.atomic_write_text(C.P("state", "chunk_run.pid"), str(os.getpid()))
    all_pmcids = list(enumerate_downloaded())
    todo = [p for p in all_pmcids if not CH.is_chunked(p)]
    C.log("INFO", "chunk_run", "loading groups+aliases in parent (shared via fork COW)...")
    _G["prop"], _G["mols"] = CH.load_groups()
    _G["aliases"] = CH.load_aliases()
    C.log("INFO", "chunk_run", "chunking %d articles (%d already done) on %d procs"
          % (len(todo), len(all_pmcids) - len(todo), PROCS))
    counts = _counts0()
    counts["cached"] = len(all_pmcids) - len(todo)
    t0 = time.time()
    with mp.Pool(PROCS) as pool:
        for i, (status, pmcid, npar, ntab, inp, inm) in enumerate(
                pool.imap_unordered(_work, todo, chunksize=40), 1):
            if status == "ok":
                counts["articles_parsed"] += 1
                counts["paragraph_passages"] += npar
                counts["table_passages"] += ntab
                tot = npar + ntab
                if inp:
                    counts["property_articles"] += 1
                    counts["property_passages"] += tot
                if inm:
                    counts["molecule_articles"] += 1
                    counts["molecule_passages"] += tot
                if inp and inm:
                    counts["both_articles"] += 1
                    counts["both_passages"] += tot
            elif status == "err":
                counts["errors"] += 1
            elif status == "missing":
                counts["missing"] += 1
            if i % 2000 == 0:
                _write(counts, len(todo), i, t0)
                C.log("INFO", "chunk_run", "%d/%d articles; %d passages (%.0f art/s)"
                      % (i, len(todo), counts["paragraph_passages"] + counts["table_passages"],
                         i / max(time.time() - t0, 1e-6)))
    _write(counts, len(todo), len(todo), t0, final=True)
    C.log("INFO", "chunk_run", "CHUNKING DONE: %d articles, %d paragraphs, %d tables (property_art=%d molecule_art=%d both=%d)"
          % (counts["articles_parsed"], counts["paragraph_passages"], counts["table_passages"],
             counts["property_articles"], counts["molecule_articles"], counts["both_articles"]))
    return counts

if __name__ == "__main__":
    run()
