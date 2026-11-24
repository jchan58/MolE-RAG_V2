"""Iterative A/B/C evidence framework, Qwen-32B, BACE. Validation first then test (arg). Label-isolated, resumable.
A: facts+15desc + 5 enriched ecfp2 refs + task def -> assessment+prediction.
B: up to 3 reviews (no new evidence), reuse A.
C: up to 3 rounds: ask 1 question (4 types) -> route (compound/task cohere retrieval | structural tool | task-def) -> assess -> stop/continue.
Bounded: rounds<=3, lit searches<=3, passages<=3/search, tool checks<=2. No target labels in prompts (ref labels shown)."""
import csv,json,re,time,math,urllib.request,sys,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import os
_HERE=Path(__file__).resolve().parent; _REPO=_HERE.parent.parent
sys.path.insert(0,str(_HERE.parent)); sys.path.insert(0,str(_HERE))  # framework/ (model_client) + this dir (c3tool)
import c3tool
from accfg import AccFG  # pip install accfg>=0.0.9
try:
    from dotenv import load_dotenv; load_dotenv(_REPO/'.env')
except Exception: pass
from model_client import load_models, chat as _chat, embed as _embed
BASE=_HERE                                                     # holds packets_bace.json
RUNS=Path(os.environ.get('MOLERAG_RUNS', _REPO/'runs'))/'bace'
SPLIT=sys.argv[1] if len(sys.argv)>1 else 'valid'; W=int(sys.argv[2]) if len(sys.argv)>2 else 4
MODEL=sys.argv[3] if len(sys.argv)>3 else 'qwen3_32b'
_MODELS=load_models(_REPO); MCFG=_MODELS[MODEL]; ECFG=_MODELS['embedder']
PARTS=RUNS/f'{MODEL}_{SPLIT}'; PARTS.mkdir(parents=True,exist_ok=True)
_calls=[0]; _lock=threading.Lock()
# bace C-route retrieval pool (pool.json + pool_emb.json). NOTE: the original pool lived in an
# ephemeral scratchpad and is not recoverable; regenerate it from the bace cards (see docs/REPRODUCIBILITY.md).
POOLD=Path(os.environ.get('MOLERAG_BACE_POOL', _REPO/'data'/'bace_pool'))
DESC=["NumHeteroatoms","BertzCT","HeavyAtomCount","Chi0","Chi1","SlogP_VSA2","MaxEStateIndex","MaxAbsEStateIndex","HeavyAtomMolWt","NumValenceElectrons","ExactMolWt","MolWt","LabuteASA","SMR_VSA1","Chi0n"]
GLOSS="(BertzCT=complexity; Chi0/Chi1/Chi0n=connectivity indices; SlogP_VSA2/SMR_VSA1/LabuteASA=VSA surface-area descriptors; MaxEStateIndex/MaxAbsEStateIndex=electrotopological state; NumValenceElectrons; the rest are size/weight/heteroatom counts). Descriptor importance does NOT establish that increasing a value increases activity."
TASKDEF=("TASK: predict human BACE-1 (beta-secretase 1) inhibition. Dataset label convention: 1=ACTIVE inhibitor, 0=INACTIVE. "
 "Report probability_active = probability of the ACTIVE class. General drug-like properties alone do NOT establish activity. "
 "Training-reference labels are provided; the target's label is withheld. Do not invent assay thresholds or conditions.")
def bcall(prompt,mt=1500):
    txt,st=_chat(prompt,MCFG,mt)
    if st=="ok":
        with _lock: _calls[0]+=1
    return txt,st
def _san(s):
    s=s.replace('，',',').replace('：',':').replace('“','"').replace('”','"')
    s=re.sub(r'\\(?!["\\/bfnrtu])',r'\\\\',s);s=re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)',r'\1"\2"\3',s);s=re.sub(r',(\s*[}\]])',r'\1',s);return s
def escf(t):
    o=[];ins=False;e=False
    for c in t:
        if e:o.append(c);e=False;continue
        if c=='\\':o.append(c);e=True;continue
        if c=='"':ins=not ins;o.append(c);continue
        if ins and c in '\n\r\t':o.append({'\n':'\\n','\r':'\\r','\t':'\\t'}[c]);continue
        o.append(c)
    return ''.join(o)
def xj(raw):
    t=escf(re.sub(r'<think>.*?</think>','',raw or '',flags=re.S).replace('```json','').replace('```',''));i=t.find('{')
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
    if end is not None:
        for cc in (t[i:end+1],_san(t[i:end+1])):
            try:return json.loads(cc)
            except:pass
    frag=(t[i:]+('"' if ins else ''));frag=re.sub(r',\s*$','',frag.rstrip())+''.join(reversed(st))
    for cc in (frag,_san(frag)):
        try:return json.loads(cc)
        except:pass
    return None
# ---- references (ecfp2 top-5 from train) ----
pk=json.load(open(BASE/'packets_bace.json'))
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
    import json as _j
    q=fp(p['standardized_SMILES']); tg=_groups(p['standardized_SMILES']); td=p['descriptors']
    from rdkit.Chem import DataStructs as _DS
    sims=sorted([(_DS.TanimotoSimilarity(q,tf),tp) for tp,tf in TRAIN if tf is not None and tp['standardized_SMILES']!=p['standardized_SMILES']],key=lambda x:-x[0])[:5]
    DESC=["NumHeteroatoms","BertzCT","HeavyAtomCount","Chi0","Chi1","SlogP_VSA2","MaxEStateIndex","MaxAbsEStateIndex","HeavyAtomMolWt","NumValenceElectrons","ExactMolWt","MolWt","LabuteASA","SMR_VSA1","Chi0n"]
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
# ---- cohere retrieval pool ----
pool=json.load(open(POOLD/'pool.json')); emb=json.load(open(POOLD/'pool_emb.json')); pool=[x for x in pool if x['pid'] in emb]
def eq(t):
    v=_embed(t,ECFG); return v[0] if v else None
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
    # separate tool EXECUTION from question RESOLUTION; invoke aminoquinoline tool only where it applies
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
def P_A(tgt,refs):
    rb="\n".join(fact_block(tp,True,int(l),s) for (l,tp) in [(p['label'],p) for s,p in []] ) # placeholder
    return None
# build prompts inline in worker (closures need tgt/refs)
def run_mol(p):
    f=PARTS/f"{SPLIT}_{p['idx']:04d}.json"
    if f.exists(): return
    tgt_facts=fact_block(p); refs=refs_for(p)
    refb="\n".join(fact_block(tp,True,tp['label'],s) for s,tp in refs)
    DESCGL="DESCRIPTOR GLOSSARY: "+GLOSS
    base=f"{TASKDEF}\n{DESCGL}\n\nTARGET FACTS:\n{tgt_facts}\n\nFIVE SIMILAR TRAINING REFERENCES (labels are measured outcomes):\n{refb}\n"
    # A
    import os as _os
    diffblk=diff_block(p)
    base=base+"\nEXPLICIT TARGET-REFERENCE DIFFERENCES (descriptor + functional-group; NOT verified-structural):\n"+diffblk+"\n"
    V4=str(RUNS/f'qwen_v4_{SPLIT}'/f"{SPLIT}_{p['idx']:04d}.json")  # optional A/C cache; absent on a fresh clone -> recompute
    reuseC=None
    if MODEL=='qwen3_32b' and _os.path.exists(V4):
        v4=json.load(open(V4)); ad={"raw":v4['A_diff']['raw'],"parsed":v4['A_diff']['parsed']}; A=v4['A_diff']['parsed'] or {}; reuseC=v4.get('C'); reused=True
    else:
        ad_raw,_=bcall(base+"\nGive a concise assessment, key supporting facts, uncertainties, and a prediction.\n"+SCHEMA); A=xj(ad_raw) or {}; ad={"raw":ad_raw,"parsed":A}; reused=False
    rec={"idx":p['idx'],"split":SPLIT,"model":MODEL,"label_SEALED":p['label'],"condition":"descriptor_FG_differences","reused_A_C_from_v4":reused,"A_diff":ad,"B":[],"C":{"rounds":[]}}
    # C-only run: no B reviews (A_diff is C's starting assessment)
    if reuseC is not None:
        rec["C"]=reuseC
        rec["calls_so_far"]=_calls[0]; json.dump(rec,open(f,'w')); return
    a_line=f"\nYOUR INITIAL ASSESSMENT: {A.get('justification','')} | initial probability_active={A.get('probability_active')}\n"
    # C: up to 3 evidence rounds
    searches=0; toolchecks=0; carried=""
    curprob=A.get('probability_active')
    for rnd in range(3):
        qp=(base+a_line+(f"\nEVIDENCE SO FAR:\n{carried}\n" if carried else "")+"\n"+TOOL_CATALOG+"\n"
            "\nIdentify the MOST IMPORTANT unresolved, prediction-relevant question. Types: task_understanding | scientific_relationship | compound_evidence | structural_check. "
            "CONTINUE ONLY IF both hold: (1) a specific unresolved question whose answer could change the prediction, AND (2) an AVAILABLE route capable of addressing it -- literature retrieval (for task/scientific/compound questions) or the aminoquinoline_c3_checker (ONLY if the question concerns a 2-aminoquinoline C-3 side chain). "
            "If no such question+route pair exists, STOP. Do NOT continue merely because the probability has not stabilized.\n"
            'Output ONLY JSON: {"stop":false,"question":"...","question_type":"task_understanding|scientific_relationship|compound_evidence|structural_check","route":"literature|aminoquinoline_c3_checker|none","route_available":true,"why_matters":"..."}')
        q_raw,_=bcall(qp); Q=xj(q_raw) or {}
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
        # assess
        evtxt=json.dumps(ev)[:2600]
        ap=(base+a_line+f"\nQUESTION ({qtype}): {Q['question']}\nEVIDENCE RETURNED ({evstat}):\n{evtxt}\n"
            "Assess: does the source support the statement, and does it APPLY to THIS target? An unanswered search is NOT evidence of inactivity; a potency value for an unrelated compound does not answer a target-specific question. Update probability only if warranted.\n"
            "Note: a tool result such as \'not a 2-aminoquinoline\' is a valid STRUCTURAL FACT about the target but does NOT by itself resolve an unrelated question. Record tool_execution_status (did the tool run / apply) SEPARATELY from question_answer_status (did the evidence answer THIS question).\n"
            'Output ONLY JSON: {"tool_execution_status":"executed|not_applicable|none","question_answer_status":"answered|partially_answered|not_answered|unresolved","answer_status":"answered|partially_answered|not_answered","evidence_subject":"target|reference|related_compound|general_class|unresolved","applicability_to_target":"...","new_information_added":true,"probability_active":0.0,"next_action":"continue|stop","stop_reason":"...","justification":"..."}')
        as_raw,_=bcall(ap); AS=xj(as_raw) or {}
        if isinstance(AS.get('probability_active'),(int,float)): curprob=AS['probability_active']
        carried+=f"\n[round{rnd+1}] Q({qtype}): {Q['question'][:120]} -> {AS.get('answer_status')}, subj={AS.get('evidence_subject')}, prob={curprob}"
        rec["C"]["rounds"].append({"ask":{"raw":q_raw,"parsed":Q},"evidence":ev,"evidence_type":evstat,"assess":{"raw":as_raw,"parsed":AS}})
        if AS.get('next_action')=='stop': break
    # final C prediction = last round's assess (or A if no rounds)
    lastC=None
    for r in reversed(rec["C"]["rounds"]):
        if r.get('assess',{}).get('parsed'): lastC=r['assess']['parsed']; break
    rec["C"]["final"]=lastC or A
    rec["calls_so_far"]=_calls[0]
    json.dump(rec,open(f,'w'))
rows=[p for p in pk[SPLIT] if not p.get('error')]
todo=[p for p in rows if not (PARTS/f"{SPLIT}_{p['idx']:04d}.json").exists()]
print(f"framework {SPLIT}: {len(todo)}/{len(rows)} molecules todo",flush=True)
def _safe(p):
    try: return run_mol(p)
    except RuntimeError as e: print(f"[skip idx={p.get('idx')}: {str(e)[:120]}]",flush=True)
with ThreadPoolExecutor(max_workers=W) as ex: list(ex.map(_safe,todo))
print(f"framework {SPLIT} DONE, total qwen calls {_calls[0]}",flush=True)
