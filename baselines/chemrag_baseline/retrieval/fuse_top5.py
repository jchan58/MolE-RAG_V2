"""RRF-fuse BM25 + Contriever hit lists, take top-5 corpus ids per query (ChemRAG combined retriever).
score(id) = sum_retrievers 1/(RRF_K + rank), RRF_K=60. Writes fused_top5.jsonl {topic_id, top5:[id,...]}
and needed_ids.json (union for content fetch)."""
import json,os
HERE=os.path.dirname(os.path.abspath(__file__))
RRF_K=60; TOP=5
def load(p):
    d={}
    for l in open(p):
        r=json.loads(l); d[r["topic_id"]]={h["id"]:h["rank"] for h in r["hits"]}
    return d
def main():
    bm=load(f"{HERE}/bm25_hits.jsonl"); de=load(f"{HERE}/dense_hits.jsonl")
    tids=list(dict.fromkeys(list(bm)+list(de)))
    need=set(); out=open(f"{HERE}/fused_top5.jsonl","w")
    for tid in tids:
        sc={}
        for src in (bm.get(tid,{}),de.get(tid,{})):
            for i,rk in src.items(): sc[i]=sc.get(i,0.0)+1.0/(RRF_K+rk)
        top=[i for i,_ in sorted(sc.items(),key=lambda x:-x[1])[:TOP]]
        need.update(top)
        out.write(json.dumps({"topic_id":tid,"top5":top})+"\n")
    out.close()
    json.dump(sorted(need),open(f"{HERE}/needed_ids.json","w"))
    print(f"fused {len(tids)} queries | unique ids to fetch: {len(need)}")
if __name__=="__main__": main()
