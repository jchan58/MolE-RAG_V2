"""BM25 retrieval over the ChemRAG corpus for all molecule queries (plainqafact env + Java 21).
Uses LuceneSearcher.batch_search (one warm searcher, many threads) — fast vs exp44's per-query loop.
Resumable: skips topic_ids already in bm25_hits.jsonl. k=60 (POOL).
Run: PYTHONNOUSERSITE=1 JAVA_HOME=... <plainqafact python> bm25_run.py"""
import json,time,sys,os,argparse
from pyserini.search.lucene import LuceneSearcher
HERE=os.path.dirname(os.path.abspath(__file__))
QF=f"{HERE}/../queries/all_queries.jsonl"
INDEX="/workspace/MolE-RAG/external/ChemRAG/index/bm25/bm25"
K=60; CHUNK=500
_ap=argparse.ArgumentParser(); _ap.add_argument("--shard",type=int,default=0); _ap.add_argument("--nshards",type=int,default=1)
_ap.add_argument("--threads",type=int,default=64); _ap.add_argument("--minutes",type=float,default=52); _A=_ap.parse_args()
THREADS=_A.threads; BUDGET=_A.minutes
OUT=f"{HERE}/bm25_hits.jsonl" if _A.nshards==1 else f"{HERE}/bm25_hits_shard{_A.shard}.jsonl"
def main():
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["topic_id"])
            except: pass
    allq=[json.loads(l) for l in open(QF)]
    qs=[q for i,q in enumerate(allq) if i%_A.nshards==_A.shard and q["topic_id"] not in done]
    print(f"BM25: {len(qs)} pending ({len(done)} done)",flush=True)
    if not qs: print("BM25 ALL DONE"); return
    t_open=time.time(); s=LuceneSearcher(INDEX); print(f"searcher open {time.time()-t_open:.0f}s",flush=True)
    t0=time.time(); n=0
    with open(OUT,"a") as f:
        for i in range(0,len(qs),CHUNK):
            if (time.time()-t0)/60>BUDGET:
                print(f"TIME BUDGET {BUDGET}min reached; {n} done this job",flush=True); break
            batch=qs[i:i+CHUNK]
            res=s.batch_search([q["query"] for q in batch],[q["topic_id"] for q in batch],k=K,threads=THREADS)
            for q in batch:
                hits=res.get(q["topic_id"],[])
                rows=[{"id":int(h.docid),"rank":r+1,"score":float(h.score)} for r,h in enumerate(hits)]
                f.write(json.dumps({"topic_id":q["topic_id"],"hits":rows})+"\n")
            f.flush(); n+=len(batch)
            el=time.time()-t0
            print(f"  {n}/{len(qs)} | {el:.0f}s | {el/n:.2f}s/q | eta {el/n*(len(qs)-n)/60:.0f}min",flush=True)
    print(f"BM25 DONE {n} queries in {(time.time()-t0)/60:.1f}min")
if __name__=="__main__": main()
