"""Contriever dense retrieval over the ChemRAG corpus for all molecule queries (pls_env + GPU).
Encodes all queries with facebook/contriever-msmarco (mean-pool), then a single GPU block-scan over the
6 fp16 embedding chunks (inner product, matching exp44 / IndexFlatIP), keeping top-30 corpus ids per query.
Run: PYTHONNOUSERSITE=1 <pls_env python> dense_run.py"""
import json,os,time,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
HERE=os.path.dirname(os.path.abspath(__file__))
QF=f"{HERE}/../queries/all_queries.jsonl"
OUT=f"{HERE}/dense_hits.jsonl"
QEMB=f"{HERE}/query_emb.npy"
IDX="/workspace/contriever_index"
MODEL="/workspace/models/hub/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa"
DIM=768; TOPK=30; BLOCK=50000
dev="cuda" if torch.cuda.is_available() else "cpu"
def mean_pool(last,mask):
    m=mask.unsqueeze(-1).float(); return (last*m).sum(1)/m.sum(1).clamp(min=1e-9)
def encode_queries(qs):
    if os.path.exists(QEMB):
        Q=np.load(QEMB)
        if Q.shape[0]==len(qs): print("reuse query_emb",Q.shape); return Q
    tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModel.from_pretrained(MODEL).to(dev).eval()
    embs=[]; B=256
    with torch.no_grad():
        for i in range(0,len(qs),B):
            b=[q["query"] for q in qs[i:i+B]]
            enc=tok(b,padding=True,truncation=True,max_length=256,return_tensors="pt").to(dev)
            out=model(**enc); e=mean_pool(out.last_hidden_state,enc["attention_mask"])
            embs.append(e.float().cpu().numpy())
            if i%2560==0: print(f"  encode {i}/{len(qs)}",flush=True)
    Q=np.concatenate(embs,0).astype(np.float32); np.save(QEMB,Q); print("encoded",Q.shape); return Q
def main():
    qs=[json.loads(l) for l in open(QF)]; nq=len(qs)
    Q=encode_queries(qs)                      # nq x 768 fp32
    Qt=torch.from_numpy(Q).to(dev).half()     # nq x 768 fp16 (tensor-core matmul)
    CKPT=f"{HERE}/dense_ckpt.pt"; start_chunk=0
    best_val=torch.full((nq,TOPK),-65000.0,device=dev,dtype=torch.float16)
    best_id=torch.full((nq,TOPK),-1,dtype=torch.long,device=dev)
    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=dev); best_val=ck["val"].to(dev); best_id=ck["id"].to(dev); start_chunk=ck["next_chunk"]
        print(f"resume from chunk {start_chunk}",flush=True)
    t0=time.time()
    for c in range(start_chunk,6):
        ids=np.array(json.load(open(f"{IDX}/chunk_{c}_ids.json")),dtype=np.int64)
        n=len(ids); fp=np.memmap(f"{IDX}/chunk_{c}_embeddings.npy",dtype=np.float16,mode="r",shape=(n,DIM))
        ids_t=torch.from_numpy(ids).to(dev)
        for s in range(0,n,BLOCK):
            e=s+min(BLOCK,n-s)
            blk=torch.from_numpy(np.ascontiguousarray(fp[s:e])).to(dev)            # b x 768 fp16
            sc=blk@Qt.T                                                            # b x nq fp16 (tensor cores)
            k=min(TOPK,e-s)
            v,ix=torch.topk(sc,k,dim=0)                                            # k x nq (ix in 0..e-s-1)
            cand_id=ids_t[s+ix]                                                    # k x nq global corpus ids
            # merge with running best
            allv=torch.cat([best_val,v.T],1)        # nq x (TOPK+k)
            allid=torch.cat([best_id,cand_id.T],1)  # nq x (TOPK+k)
            nv,nix=torch.topk(allv,TOPK,dim=1)
            best_val=nv; best_id=torch.gather(allid,1,nix)
        torch.save({"val":best_val.cpu(),"id":best_id.cpu(),"next_chunk":c+1},CKPT)
        print(f"  chunk {c} done n={n} | {time.time()-t0:.0f}s | ckpt saved",flush=True)
    bv=best_val.cpu().numpy(); bi=best_id.cpu().numpy()
    with open(OUT,"w") as f:
        for i,q in enumerate(qs):
            order=np.argsort(-bv[i])
            rows=[{"id":int(bi[i][j]),"rank":r+1,"score":float(bv[i][j])} for r,j in enumerate(order) if bi[i][j]>=0]
            f.write(json.dumps({"topic_id":q["topic_id"],"hits":rows})+"\n")
    print(f"DENSE DONE {nq} queries in {(time.time()-t0)/60:.1f}min")
if __name__=="__main__": main()
