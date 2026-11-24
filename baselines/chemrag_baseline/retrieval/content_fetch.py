"""Fetch verbatim contents+source for the union of top-5 ids, one sequential pass over the 6 corpus chunks.
Writes id_contents.jsonl {id, contents, source}. Then assembles fused_docs.jsonl joining fused_top5 with text."""
import json,os
HERE=os.path.dirname(os.path.abspath(__file__))
CORPUS="/workspace/MolE-RAG/external/ChemRAG/corpus/chunks"
def main():
    need=set(json.load(open(f"{HERE}/needed_ids.json")))
    print(f"need {len(need)} ids",flush=True)
    got={}
    for c in range(6):
        p=f"{CORPUS}/chunk_{c}.jsonl"
        if not os.path.exists(p): continue
        with open(p) as f:
            for l in f:
                try: r=json.loads(l)
                except: continue
                if r["id"] in need:
                    got[r["id"]]={"contents":r.get("contents",""),"source":r.get("source","")}
                    if len(got)==len(need): break
        print(f"  chunk {c}: have {len(got)}/{len(need)}",flush=True)
        if len(got)==len(need): break
    with open(f"{HERE}/id_contents.jsonl","w") as f:
        for i,v in got.items(): f.write(json.dumps({"id":i,**v})+"\n")
    # assemble: join fused_top5 with contents
    with open(f"{HERE}/fused_docs.jsonl","w") as out:
        for l in open(f"{HERE}/fused_top5.jsonl"):
            r=json.loads(l)
            docs=[{"id":i,"source":got.get(i,{}).get("source",""),"contents":got.get(i,{}).get("contents","")} for i in r["top5"]]
            out.write(json.dumps({"topic_id":r["topic_id"],"docs":docs})+"\n")
    print(f"content fetch DONE: {len(got)} ids; fused_docs.jsonl written")
if __name__=="__main__": main()
