"""SageMaker entry: ChemDFM-v1.0-13B on the BACE iterative evidence framework (conditions A and C).

Faithful port of molerag_reasoning/BACE/framework_final/run_final.py -- IDENTICAL prompts, references
(ecfp2 top-5), 15 descriptors, target-reference differences, Cohere retrieval pool, aminoquinoline
structural tool, and bounded C loop (<=3 rounds, <=3 lit searches, <=2 tool checks). The ONLY change
is the LLM call: instead of a Bedrock/OpenAI bcall, we generate locally with ChemDFM
(LlamaForCausalLM, dialogue format `[Round 0]\nHuman: {prompt}\nAssistant:`, greedy).

Each molecule's record stores A_diff (== condition A: facts+refs+differences -> assessment+prediction)
AND C (== the iterative framework). ChemDFM is LLaMA-1-13B with a 2048-token context; the framework
prompts are much longer, so inputs are head+tail truncated to fit (documented) and outputs are
expected to be frequently malformed -- this run is a best-effort "try both A and C on ChemDFM".

Cohere retrieval still goes over the network with the Bedrock bearer token (env BEDROCK_TOKEN);
bearer-token auth is independent of the SageMaker execution role. Per-molecule results are streamed to
s3://<bucket>/chemdfm_fw/parts/test_XXXX.json (restart-safe: done molecules skipped on resume)."""
import os, sys, json, re, time, math
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from transformers import LlamaTokenizer, LlamaForCausalLM
import urllib.request
import boto3
from accfg import AccFG

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import c3tool
MODEL_NAME = os.environ.get("MODEL_NAME", "OpenDFM/ChemDFM-v1.0-13B")
SPLIT = os.environ.get("SPLIT", "test")
S3_BUCKET = os.environ.get("S3_BUCKET", "molerag")
S3_PARTS = os.environ.get("S3_PARTS", "chemdfm_fw/parts")
BEDROCK_TOKEN = os.environ.get("BEDROCK_TOKEN", "")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "420"))
LOCAL_PARTS = "/opt/ml/model/parts"; os.makedirs(LOCAL_PARTS, exist_ok=True)
s3 = boto3.client("s3", region_name="us-east-1")

DESC=["NumHeteroatoms","BertzCT","HeavyAtomCount","Chi0","Chi1","SlogP_VSA2","MaxEStateIndex","MaxAbsEStateIndex","HeavyAtomMolWt","NumValenceElectrons","ExactMolWt","MolWt","LabuteASA","SMR_VSA1","Chi0n"]
GLOSS="(BertzCT=complexity; Chi0/Chi1/Chi0n=connectivity indices; SlogP_VSA2/SMR_VSA1/LabuteASA=VSA surface-area descriptors; MaxEStateIndex/MaxAbsEStateIndex=electrotopological state; NumValenceElectrons; the rest are size/weight/heteroatom counts). Descriptor importance does NOT establish that increasing a value increases activity."
TASKDEF=("TASK: predict human BACE-1 (beta-secretase 1) inhibition. Dataset label convention: 1=ACTIVE inhibitor, 0=INACTIVE. "
 "Report probability_active = probability of the ACTIVE class. General drug-like properties alone do NOT establish activity. "
 "Training-reference labels are provided; the target's label is withheld. Do not invent assay thresholds or conditions.")

# ---------------- JSON extraction (identical to run_final.xj) ----------------
def xj(raw):
    t=re.sub(r'<think>.*?</think>','',raw or '',flags=re.S).replace('```json','').replace('```','');i=t.find('{')
    if i<0:return None
    st=[];ins=False;e=False;end=None
    for j in range(i,len(t)):
        c=t[j]
        if e:e=False;continue
        if c=='\\':e=True;continue
        if c=='"':ins=not ins;continue
        if ins:continue
        if c in '{[':st.append('}' if c=='{' else ']')
        elif c in '}]':
            if st:st.pop()
            if not st:end=j;break
    frag=t[i:(end+1 if end else len(t))]
    for cc in (frag,re.sub(r',(\s*[}\]])',r'\1',frag)):
        try:return json.loads(cc)
        except:pass
    return None

# ---------------- references + facts (identical to run_final) ----------------
pk=json.load(open(os.path.join(CODE,'packets_bace.json')))
def fp(s):
    m=Chem.MolFromSmiles(s); return AllChem.GetMorganFingerprintAsBitVect(m,1,2048) if m else None
TRAIN=[(p,fp(p['standardized_SMILES'])) for p in pk['train'] if not p.get('error')]
_afg=AccFG(lite=True); _afgc={}
def accfg(s):
    if s in _afgc: return _afgc[s]
    try: fg,_=_afg.run(s,show_atoms=True,show_graph=True); r=", ".join(f"{k} x{len(v)}" for k,v in fg.items()) or "none"
    except Exception: r="unavailable"
    _afgc[s]=r; return r
def _groups(s):
    try: g,_=_afg.run(s,show_atoms=True,show_graph=True); return {k:len(v) for k,v in g.items()}
    except Exception: return {}
def diff_block(p):
    q=fp(p['standardized_SMILES']); tg=_groups(p['standardized_SMILES']); td=p['descriptors']
    sims=sorted([(DataStructs.TanimotoSimilarity(q,tf),tp) for tp,tf in TRAIN if tf is not None and tp['standardized_SMILES']!=p['standardized_SMILES']],key=lambda x:-x[0])[:5]
    lines=[]
    for sdv,tp in sims:
        rg=_groups(tp['standardized_SMILES']); rd=tp['descriptors']
        onlyt=[f"{k}x{tg[k]}" for k in tg if k not in rg]; onlyr=[f"{k}x{rg[k]}" for k in rg if k not in tg]
        dd=sorted(((k,round(td[k]-rd[k],2)) for k in DESC if td[k] is not None and rd[k] is not None),key=lambda x:-abs(x[1]))[:6]
        lines.append(f"  vs ref(outcome={'ACTIVE' if tp['label']==1 else 'INACTIVE'},Tani={round(sdv,2)}): FG only-in-target={onlyt or '[]'}; FG only-in-ref={onlyr or '[]'}; top|Δdescriptor|(target-ref)={dd}")
    return "\n".join(lines)
def refs_for(p):
    q=fp(p['standardized_SMILES']); sims=[]
    for tp,tf in TRAIN:
        if tf is None or tp['standardized_SMILES']==p['standardized_SMILES']: continue
        sims.append((DataStructs.TanimotoSimilarity(q,tf),tp))
    sims.sort(key=lambda x:-x[0]); return sims[:5]
def fact_block(p,is_ref=False,lab=None,sim=None):
    d=p['descriptors']; dd="; ".join(f"{n}={d[n]}" for n in DESC)
    head=(f"id={('ref' if is_ref else 'TARGET')}; standardized_SMILES={p['standardized_SMILES']}; formula={p['molecular_formula']}; "
          f"AccFG={accfg(p['standardized_SMILES'])}; stereocenters={p['n_stereocenters']} (unassigned {p['n_unassigned_stereo']})")
    if is_ref: head=f"[ref outcome={'ACTIVE' if lab==1 else 'INACTIVE'}; Tanimoto={round(sim,3)}] "+head
    return head+f"; descriptors: {dd}"

# ---------------- cohere retrieval (network, bearer token) ----------------
pool=json.load(open(os.path.join(CODE,'pool.json'))); emb=json.load(open(os.path.join(CODE,'pool_emb.json'))); pool=[x for x in pool if x['pid'] in emb]
def eq(t):
    for a in range(6):
        try:
            req=urllib.request.Request("https://bedrock-runtime.us-east-1.amazonaws.com/model/cohere.embed-multilingual-v3/invoke",data=json.dumps({"texts":[t],"input_type":"search_query","truncate":"END"}).encode(),headers={"Authorization":"Bearer "+BEDROCK_TOKEN,"Content-Type":"application/json","Accept":"application/json"})
            d=json.load(urllib.request.urlopen(req,timeout=90));e=d["embeddings"];return (e["float"] if isinstance(e,dict) else e)[0]
        except urllib.error.HTTPError as ex:
            if ex.code==429: time.sleep(4+4*a); continue
            return None
        except: time.sleep(3+3*a)
    return None
def cos(a,b): s=sum(x*y for x,y in zip(a,b));return s/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b))+1e-9)
def retrieve(q,k=3):
    v=eq(q)
    if v is None: return [],"query_embed_failed"
    top=sorted(((cos(v,emb[x['pid']]),x) for x in pool),key=lambda z:-z[0])[:k]
    return [{"src":f"PMC{x['art']}","key":x['aid'],"cos":round(s,3),"text":x['q']} for s,x in top],"ok"
TOOL_CATALOG=("SUPPORTED STRUCTURAL TOOLS (the ONLY implemented checks):\n"
 "  - aminoquinoline_c3_checker (input: SMILES): determines whether the molecule is a 2-aminoquinoline and, if so, whether it bears a C-3 side chain. "
 "APPLIES ONLY to 2-aminoquinolines; for any other scaffold it returns NOT_APPLICABLE and the question stays UNRESOLVED.\n"
 "No general-purpose structural checker exists. A structural question not answerable by this tool must be treated as UNRESOLVED (do not infer activity from an unanswered check).")
def struct_check(smi):
    try:
        a=c3tool.c3_analysis(smi)
        if a.get('status')=='resolved' and a.get('aminoquinoline_core_present'):
            return {"tool":"aminoquinoline_c3_checker","tool_execution_status":"executed","applicable":True,
                    "result":c3tool.gen_feedback(a),"detail":{k:a.get(k) for k in('aminoquinoline_core_present','c3_sidechain_present','c3_sidechain_smiles')}}
        return {"tool":"aminoquinoline_c3_checker","tool_execution_status":"not_applicable","applicable":False,
                "result":"The only supported structural tool (2-aminoquinoline C-3 checker) does NOT apply: this molecule is not a 2-aminoquinoline. The structural question is UNRESOLVED unless it specifically concerns a 2-aminoquinoline C-3 side chain.",
                "question_answer_status_hint":"unresolved"}
    except Exception as e: return {"tool":"aminoquinoline_c3_checker","tool_execution_status":"error","applicable":False,"result":f"tool error: {str(e)[:60]}","question_answer_status_hint":"unresolved"}
SCHEMA='Output ONLY JSON: {"prediction":"active|inactive|uncertain","probability_active":0.0,"justification":"...","unresolved_questions":[]}'

# ---------------- ChemDFM local generation ----------------
print("loading ChemDFM ...", flush=True); t0=time.time()
tok=LlamaTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model=LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto"); model.eval()
device="cuda:0" if torch.cuda.is_available() else "cpu"
CTX=getattr(model.config,"max_position_embeddings",2048)
HEAD=min(480, CTX//4)
print(f"loaded in {time.time()-t0:.0f}s; ctx={CTX}", flush=True)
_tr_marker=tok("\n...[middle of prompt truncated to fit ChemDFM context]...\n",add_special_tokens=False)["input_ids"]
_calls=[0]
def gen(prompt, max_new_tokens=MAX_NEW):
    text=f"[Round 0]\nHuman: {prompt.strip()}\nAssistant:"
    ids=tok(text,add_special_tokens=True)["input_ids"]
    budget=CTX-max_new_tokens-8; truncated=False
    if len(ids)>budget:
        truncated=True; keep_tail=budget-HEAD-len(_tr_marker)
        ids=ids[:HEAD]+_tr_marker+ids[-keep_tail:]
    enc=torch.tensor([ids],device=device); att=torch.ones_like(enc)
    with torch.no_grad():
        out=model.generate(input_ids=enc,attention_mask=att,max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,pad_token_id=(tok.eos_token_id or 0))
    new=out[0][enc.shape[1]:]; _calls[0]+=1
    return tok.decode(new,skip_special_tokens=True).strip(), truncated

# ---------------- per-molecule framework run (A_diff + C), identical flow ----------------
def part_key(idx): return f"{S3_PARTS}/{SPLIT}_{idx:04d}.json"
def done(idx):
    try: s3.head_object(Bucket=S3_BUCKET,Key=part_key(idx)); return True
    except Exception: return False
def put(idx,rec):
    body=json.dumps(rec).encode()
    open(os.path.join(LOCAL_PARTS,f"{SPLIT}_{idx:04d}.json"),"wb").write(body)
    s3.put_object(Bucket=S3_BUCKET,Key=part_key(idx),Body=body)

def run_mol(p):
    idx=p['idx']
    if done(idx): return "skip"
    tgt_facts=fact_block(p); refs=refs_for(p)
    refb="\n".join(fact_block(tp,True,tp['label'],s) for s,tp in refs)
    base=f"{TASKDEF}\nDESCRIPTOR GLOSSARY: {GLOSS}\n\nTARGET FACTS:\n{tgt_facts}\n\nFIVE SIMILAR TRAINING REFERENCES (labels are measured outcomes):\n{refb}\n"
    base=base+"\nEXPLICIT TARGET-REFERENCE DIFFERENCES (descriptor + functional-group; NOT verified-structural):\n"+diff_block(p)+"\n"
    # A (== A_diff)
    ad_raw,ad_tr=gen(base+"\nGive a concise assessment, key supporting facts, uncertainties, and a prediction.\n"+SCHEMA)
    A=xj(ad_raw) or {}
    rec={"idx":idx,"split":SPLIT,"model":"chemdfm_v1_13b","label_SEALED":p['label'],"condition":"descriptor_FG_differences",
         "context_truncated":bool(ad_tr),"A_diff":{"raw":ad_raw,"parsed":A},"C":{"rounds":[]}}
    a_line=f"\nYOUR INITIAL ASSESSMENT: {A.get('justification','')} | initial probability_active={A.get('probability_active')}\n"
    searches=0; toolchecks=0; carried=""; curprob=A.get('probability_active')
    for rnd in range(3):
        qp=(base+a_line+(f"\nEVIDENCE SO FAR:\n{carried}\n" if carried else "")+"\n"+TOOL_CATALOG+"\n"
            "\nIdentify the MOST IMPORTANT unresolved, prediction-relevant question. Types: task_understanding | scientific_relationship | compound_evidence | structural_check. "
            "CONTINUE ONLY IF both hold: (1) a specific unresolved question whose answer could change the prediction, AND (2) an AVAILABLE route capable of addressing it -- literature retrieval (for task/scientific/compound questions) or the aminoquinoline_c3_checker (ONLY if the question concerns a 2-aminoquinoline C-3 side chain). "
            "If no such question+route pair exists, STOP. Do NOT continue merely because the probability has not stabilized.\n"
            'Output ONLY JSON: {"stop":false,"question":"...","question_type":"task_understanding|scientific_relationship|compound_evidence|structural_check","route":"literature|aminoquinoline_c3_checker|none","route_available":true,"why_matters":"..."}')
        q_raw,_=gen(qp); Q=xj(q_raw) or {}
        route=Q.get('route','literature'); ra=Q.get('route_available',True)
        if Q.get('stop') or not Q.get('question') or route=='none' or ra is False:
            rec["C"]["rounds"].append({"ask":{"raw":q_raw,"parsed":Q},"stopped":True,"stop_reason":"no unresolved question with an available capable route"}); break
        qtype=Q.get('question_type','scientific_relationship'); ev=None; evstat=None
        if route=='aminoquinoline_c3_checker' and toolchecks<2:
            ev=struct_check(p['standardized_SMILES']); toolchecks+=1; evstat="tool"
        elif searches<3:
            passages,rs=retrieve(Q['question'],3); searches+=1; ev={"passages":passages,"retrieval_status":rs}; evstat="retrieval"
        else:
            ev={"note":"search/tool budget exhausted"}; evstat="budget"
        evtxt=json.dumps(ev)[:2600]
        ap=(base+a_line+f"\nQUESTION ({qtype}): {Q['question']}\nEVIDENCE RETURNED ({evstat}):\n{evtxt}\n"
            "Assess: does the source support the statement, and does it APPLY to THIS target? An unanswered search is NOT evidence of inactivity; a potency value for an unrelated compound does not answer a target-specific question. Update probability only if warranted.\n"
            "Note: a tool result such as 'not a 2-aminoquinoline' is a valid STRUCTURAL FACT about the target but does NOT by itself resolve an unrelated question. Record tool_execution_status (did the tool run / apply) SEPARATELY from question_answer_status (did the evidence answer THIS question).\n"
            'Output ONLY JSON: {"tool_execution_status":"executed|not_applicable|none","question_answer_status":"answered|partially_answered|not_answered|unresolved","answer_status":"answered|partially_answered|not_answered","evidence_subject":"target|reference|related_compound|general_class|unresolved","applicability_to_target":"...","new_information_added":true,"probability_active":0.0,"next_action":"continue|stop","stop_reason":"...","justification":"..."}')
        as_raw,_=gen(ap); AS=xj(as_raw) or {}
        if isinstance(AS.get('probability_active'),(int,float)): curprob=AS['probability_active']
        carried+=f"\n[round{rnd+1}] Q({qtype}): {Q['question'][:120]} -> {AS.get('answer_status')}, subj={AS.get('evidence_subject')}, prob={curprob}"
        rec["C"]["rounds"].append({"ask":{"raw":q_raw,"parsed":Q},"evidence":ev,"evidence_type":evstat,"assess":{"raw":as_raw,"parsed":AS}})
        if AS.get('next_action')=='stop': break
    lastC=None
    for r in reversed(rec["C"]["rounds"]):
        if r.get('assess',{}).get('parsed'): lastC=r['assess']['parsed']; break
    rec["C"]["final"]=lastC or A
    rec["calls_so_far"]=_calls[0]
    put(idx,rec); return "ok"

rows=[p for p in pk[SPLIT] if not p.get('error')]
print(f"chemdfm framework {SPLIT}: {len(rows)} molecules", flush=True)
nok=0
for i,p in enumerate(rows):
    r=run_mol(p)
    if r=="ok": nok+=1
    if (i+1)%5==0 or r=="ok":
        print(f"  [{i+1}/{len(rows)}] idx={p['idx']} -> {r} | gens={_calls[0]}", flush=True)
print(f"DONE chemdfm framework {SPLIT}: {nok} new, total gens {_calls[0]}", flush=True)
