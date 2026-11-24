"""Frozen v4 framework (A/B/C) on ESOL/FreeSolv/BBBP/Lipo/SIDER, TEST split, 4 API models.
A: facts(15 desc)+5 ecfp2 train refs+descriptor/FG differences -> prediction (no literature).
B: up to 3 reviews, NO new evidence (matched inputs to A).
C: up to 3 rounds of question-conditioned entity->attribute retrieval over the exp70 PMC corpus:
   model asks ONE question + selects task/compound ENTITIES and ATTRIBUTES ->
   gather matching findings -> follow passage_id links to FULL passages ->
   dedup + rank passages vs the question (Cohere embed cosine) -> top-3/round -> assess/revise.
No aminoquinoline (or any) structural tool: structural questions are reported UNRESOLVED.
Target dataset label is withheld; corpus records are label-free. Regression outputs for esol/freesolv/lipo;
per-endpoint probabilities + endpoint-specific questions for sider. Resumable (one part file / target).
Usage: run_fw_multi.py <DATASET> <MODEL> [W]   (TEST split only)"""
import csv,json,re,time,math,sys,os,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs,Descriptors
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
try: from rdkit.Chem.Draw import rdMolDraw2D  # noqa
except Exception: pass
from accfg import AccFG  # pip install accfg>=0.0.9

# ---- paths & credentials: repo-relative defaults, override via .env / env ----
_HERE=Path(__file__).resolve().parent; _REPO=_HERE.parent
try:
    from dotenv import load_dotenv; load_dotenv(_REPO/'.env')
except Exception: pass
DATA_DIR=Path(os.environ.get('MOLERAG_DATA',  _REPO/'data'/'moleculenet'/'seed_0'))
CARDS_DIR=Path(os.environ.get('MOLERAG_CARDS', _REPO/'data'/'cards'))
RUNS_DIR=Path(os.environ.get('MOLERAG_RUNS',  _REPO/'runs'))
REGION=os.environ.get('AWS_REGION','us-east-1')
try: from rdkit.Chem import rdMolDescriptors as _rdm
except Exception: _rdm=None

DS=sys.argv[1]; MODEL=sys.argv[2]; W=int(sys.argv[3]) if len(sys.argv)>3 else 4
NSHARD=int(sys.argv[4]) if len(sys.argv)>4 else 1; SHARDIDX=int(sys.argv[5]) if len(sys.argv)>5 else 0
SPLIT="test"
BUNDLE=CARDS_DIR/DS
PARTS=RUNS_DIR/f'{DS}__{MODEL}'; PARTS.mkdir(parents=True,exist_ok=True)
MNET=DATA_DIR/DS
_calls=[0]; _lock=threading.Lock()

# ---------------- dataset config ----------------
CONFIG={
 "esol":{"task_type":"regression","unit":"log10 molar aqueous solubility (logS, mol/L)",
   "taskdef":"Predict the measured aqueous solubility as log10 solubility logS (mol/L); lower = less soluble. Typical range about -11 to +2. Output a single numeric logS.",
   "property":"aqueous solubility (logS)"},
 "freesolv":{"task_type":"regression","unit":"hydration free energy (kcal/mol)",
   "taskdef":"Predict the experimental hydration free energy in water (kcal/mol); more negative = more favorable hydration. Typical range about -25 to +4. Output a single numeric value in kcal/mol.",
   "property":"hydration free energy"},
 "lipo":{"task_type":"regression","unit":"logD at pH 7.4",
   "taskdef":"Predict lipophilicity as logD at pH 7.4 (ChEMBL). Typical range about -1.5 to 4.5. Output a single numeric logD.",
   "property":"lipophilicity (logD 7.4)"},
 "bbbp":{"task_type":"classification","pos_name":"BBB-penetrant","neg_name":"non-penetrant",
   "taskdef":"Predict blood-brain barrier penetration. Dataset convention: 1 = penetrant (BBB+), 0 = non-penetrant. Output probability_positive = probability the molecule is BBB-penetrant.",
   "property":"blood-brain barrier penetration"},
 "sider":{"task_type":"multilabel",
   "taskdef":"Predict side-effect occurrence across 27 MedDRA system-organ-class endpoints. For each endpoint, 1 = the drug is associated with adverse effects in that organ class. Output a probability per endpoint.",
   "property":"drug side-effect profile (27 organ classes)"},
 "clintox":{"task_type":"multilabel",
   "taskdef":"Predict two clinical-trial outcomes for the drug. CT_TOX: 1 = failed clinical trials for toxicity reasons (clinically toxic). FDA_APPROVED: 1 = approved by the FDA. Output a probability per endpoint.",
   "property":"clinical toxicity (CT_TOX) and FDA approval (FDA_APPROVED)"},
 "hiv":{"task_type":"classification","pos_name":"active (inhibits HIV replication)","neg_name":"inactive",
   "taskdef":"Predict whether the compound is ACTIVE at inhibiting HIV replication (MoleculeNet HIV screen). Dataset convention: 1 = active (confirmed/active screening result), 0 = inactive. Output probability_positive = probability the molecule is ACTIVE.",
   "property":"HIV replication inhibition (active/inactive)"},
 "tox21":{"task_type":"multilabel",
   "taskdef":"Predict activity across 12 Tox21 toxicity assays: nuclear-receptor signalling (NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma) and stress-response pathways (SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53). For each assay, 1 = the compound is ACTIVE in that assay. Output a probability per endpoint.",
   "property":"Tox21 12-assay activity profile"},
}
CFG=CONFIG[DS]; TT=CFG["task_type"]

# ---------------- model backend (provider-agnostic; configured in models.json) ----------------
from model_client import load_models, chat as _chat, embed as _embed
_MODELS=load_models(_REPO)
if MODEL not in _MODELS:
    raise SystemExit(f"model '{MODEL}' not in models.json (have: {[k for k in _MODELS if not k.startswith('_') and k!='embedder']})")
MCFG=_MODELS[MODEL]; ECFG=_MODELS["embedder"]
def bcall(prompt,mt=1600):
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
    t=escf(re.sub(r'<think>.*?</think>','',raw or '',flags=re.S).replace('```json','').replace('```','').replace('**',''));i=t.find('{')
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

# ---------------- descriptors / refs / diffs ----------------
DESC=["NumHeteroatoms","BertzCT","HeavyAtomCount","Chi0","Chi1","SlogP_VSA2","MaxEStateIndex","MaxAbsEStateIndex","HeavyAtomMolWt","NumValenceElectrons","ExactMolWt","MolWt","LabuteASA","SMR_VSA1","Chi0n"]
GLOSS="(BertzCT=complexity; Chi0/Chi1/Chi0n=connectivity indices; SlogP_VSA2/SMR_VSA1/LabuteASA=VSA surface-area descriptors; MaxEStateIndex/MaxAbsEStateIndex=electrotopological state; NumValenceElectrons; the rest are size/weight/heteroatom counts). Descriptor magnitude does NOT by itself establish the property value."
_descfns={n:getattr(Descriptors,n) for n in DESC}
def descriptors(m):
    out={}
    for n,fn in _descfns.items():
        try: v=fn(m); out[n]=round(float(v),3)
        except Exception: out[n]=None
    return out
def canon(s):
    m=Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else None
def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m,1,2048)
_afg=AccFG(lite=True); _afgc={}
def accfg(s):
    if s in _afgc: return _afgc[s]
    try: fg,_=_afg.run(s,show_atoms=True,show_graph=True); r=", ".join(f"{k} x{len(v)}" for k,v in fg.items()) or "none"
    except Exception: r="unavailable"
    _afgc[s]=r; return r
def groups(s):
    try: g,_=_afg.run(s,show_atoms=True,show_graph=True); return {k:len(v) for k,v in g.items()}
    except Exception: return {}

def load_molecules(split):
    """returns list of dicts with idx, smiles, canonical, mol, desc, fp, and target (value/label/endpoint-map)."""
    rows=[]
    f=MNET/f"{DS}_{split}.csv"
    rdr=list(csv.DictReader(open(f)))
    if TT=="multilabel":
        # melted: pivot per molecule -> {endpoint: label}
        bysmi={}
        for r in rdr:
            bysmi.setdefault(r['smiles'],{})[r['task']]=int(float(r['label']))
        items=list(bysmi.items())
        for i,(smi,epmap) in enumerate(items):
            m=Chem.MolFromSmiles(smi)
            if not m: continue
            rows.append({"idx":i,"smiles":smi,"canonical":Chem.MolToSmiles(m),"mol":m,"endpoints":epmap})
    else:
        for i,r in enumerate(rdr):
            m=Chem.MolFromSmiles(r['smiles'])
            if not m: continue
            tgt=float(r['label'])
            rows.append({"idx":i,"smiles":r['smiles'],"canonical":Chem.MolToSmiles(m),"mol":m,
                         "value":tgt if TT=="regression" else None,"label":int(tgt) if TT=="classification" else None})
    for r in rows: r["desc"]=descriptors(r["mol"]); r["fpv"]=fp(r["mol"])
    return rows

ENDPOINTS=[]
def build_train():
    global ENDPOINTS
    rows=[]
    rdr=list(csv.DictReader(open(MNET/f"{DS}_train.csv")))
    if TT=="multilabel":
        bysmi={}
        for r in rdr: bysmi.setdefault(r['smiles'],{})[r['task']]=int(float(r['label']))
        ENDPOINTS=sorted({r['task'] for r in rdr})
        for smi,epmap in bysmi.items():
            m=Chem.MolFromSmiles(smi)
            if not m: continue
            rows.append({"smiles":Chem.MolToSmiles(m),"mol":m,"endpoints":epmap,"fpv":fp(m)})
    else:
        for r in rdr:
            m=Chem.MolFromSmiles(r['smiles'])
            if not m: continue
            v=float(r['label'])
            rows.append({"smiles":Chem.MolToSmiles(m),"mol":m,"fpv":fp(m),
                         "value":v if TT=="regression" else None,"label":int(v) if TT=="classification" else None})
    for r in rows: r["desc"]=descriptors(r["mol"])
    return rows
TRAIN=build_train()

def refs_for(target):
    q=target["fpv"]; sims=[]
    for tp in TRAIN:
        if tp["smiles"]==target["canonical"]: continue
        sims.append((DataStructs.TanimotoSimilarity(q,tp["fpv"]),tp))
    sims.sort(key=lambda x:-x[0]); return sims[:5]
def ref_outcome(tp):
    if TT=="regression": return f"measured {CFG['property']} = {round(tp['value'],2)} ({CFG['unit']})"
    if TT=="classification": return f"outcome = {CFG['pos_name'] if tp['label']==1 else CFG['neg_name']}"
    # multilabel: compact positive-endpoint list
    pos=[e for e,v in tp["endpoints"].items() if v==1]
    return f"positive endpoints ({len(pos)}/{len(ENDPOINTS)}): {', '.join(pos) if pos else 'none'}"
def fact_block(target):
    d=target["desc"]; dd="; ".join(f"{n}={d[n]}" for n in DESC)
    s=target["canonical"]
    return (f"TARGET standardized_SMILES={s}; formula={_rdm.CalcMolFormula(target['mol']) if _rdm else '?'}; "
            f"heavy_atoms={target['mol'].GetNumHeavyAtoms()}; AccFG={accfg(s)}; descriptors: {dd}")
def refs_block(refs):
    out=[]
    for sim,tp in refs:
        d=tp["desc"]; dd="; ".join(f"{n}={d[n]}" for n in DESC)
        out.append(f"[ref Tanimoto={round(sim,3)}; {ref_outcome(tp)}] SMILES={tp['smiles']}; AccFG={accfg(tp['smiles'])}; descriptors: {dd}")
    return "\n".join(out)
def diff_block(target,refs):
    tg=groups(target["canonical"]); td=target["desc"]
    lines=[]
    for sim,tp in refs:
        rg=groups(tp["smiles"]); rd=tp["desc"]
        onlyt=[f"{k}x{tg[k]}" for k in tg if k not in rg]; onlyr=[f"{k}x{rg[k]}" for k in rg if k not in tg]
        dd=sorted(((k,round(td[k]-rd[k],2)) for k in DESC if td[k] is not None and rd[k] is not None),key=lambda x:-abs(x[1]))[:6]
        lines.append(f"  vs ref(Tani={round(sim,2)}): FG only-in-target={onlyt or '[]'}; FG only-in-ref={onlyr or '[]'}; top|Δdescriptor|(target-ref)={dd}")
    return "\n".join(lines)

# ---------------- C retrieval over exp70 bundle ----------------
import urllib.request
def _load_bundle():
    if not (BUNDLE/"meta.json").exists(): return None
    passages=json.load(open(BUNDLE/"passages.json"))
    fx={}; comp_attr={}  # memory-lean: keep only (attribute_id, passage_id) per entity
    for l in open(BUNDLE/"findings.jsonl"):
        try: fd=json.loads(l)
        except: continue
        eid=fd.get("entity_id"); aid=fd.get("attribute_id"); pid=fd.get("passage_id")
        if eid is None: continue
        fx.setdefault(eid,[]).append((aid,pid))
        if fd.get("entity_type")=="compound" and aid: comp_attr.setdefault(eid,set()).add(aid)
    return {"passages":passages,"_fx":fx,"_comp_attr":comp_attr,
            "entities":json.load(open(BUNDLE/"entities.json")),
            "mol_map":json.load(open(BUNDLE/"mol_map.json")),
            "meta":json.load(open(BUNDLE/"meta.json"))}
BUNDLE_DATA=_load_bundle()
def eq(t): return _embed(t,ECFG)
def cos(a,b): s=sum(x*y for x,y in zip(a,b));return s/(math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b))+1e-9)
def available_entities(target):
    """task entities (all) + this target's compound entity if it has findings."""
    ents=[{"entity_id":e,"entity_type":"task"} for e in BUNDLE_DATA["entities"]["task_entity_ids"]]
    cid=BUNDLE_DATA["mol_map"]["by_smiles"].get(target["canonical"])
    comp=None
    if cid and cid in BUNDLE_DATA["_comp_attr"]:
        comp={"entity_id":cid,"entity_type":"compound","attributes_present":sorted(BUNDLE_DATA["_comp_attr"][cid])}
        ents.append({"entity_id":cid,"entity_type":"compound"})
    return ents,comp
def do_retrieve(question,sel_entities,sel_attrs,k=3,cap=40):
    """gather findings for selected entities (optionally filtered by attributes) -> passage ids ->
       full passages -> dedup -> rank vs question -> top-k."""
    fx=BUNDLE_DATA["_fx"]; passages=BUNDLE_DATA["passages"]
    cand={}  # passage_id -> passage dict
    for eid in sel_entities:
        for aid,pid in fx.get(eid,[]):
            if sel_attrs and aid not in sel_attrs: continue
            if not pid or pid not in passages: continue
            if pid not in cand: cand[pid]=passages[pid]
            if len(cand)>=cap: break
        if len(cand)>=cap: break
    if not cand: return [],"no_findings_for_selection"
    pids=list(cand.keys()); texts=[(cand[p].get("text") or "")[:1200] for p in pids]
    embs=eq([question]+texts)
    if not embs: return [],"embed_failed"
    qv=embs[0]; scored=sorted(((cos(qv,embs[1+i]),pids[i]) for i in range(len(pids))),key=lambda x:-x[0])[:k]
    out=[{"passage_id":p,"pmcid":cand[p].get("pmcid"),"cos":round(s,3),"text":(cand[p].get("text") or "")[:600]} for s,p in scored]
    return out,"ok"

TOOL_CATALOG=("STRUCTURAL TOOLS: none are implemented for this dataset. Any question requiring a structural/substructure "
 "computation on the target CANNOT be executed and must be treated as UNRESOLVED (do not infer the property from an unrun check).")

# ---------------- task-specific schemas ----------------
if TT=="regression":
    PRED_SCHEMA='Output ONLY JSON: {"prediction_value": <number '+CFG["unit"]+'>, "justification":"...", "unresolved_questions":[]}'
    REV_SCHEMA_TAIL='"prediction_value": <number>'
    def pred_of(j):
        try: return float(j.get("prediction_value"))
        except: return None
elif TT=="classification":
    PRED_SCHEMA='Output ONLY JSON: {"prediction":"positive|negative|uncertain","probability_positive":0.0,"justification":"...","unresolved_questions":[]}'
    REV_SCHEMA_TAIL='"prediction":"positive|negative|uncertain","probability_positive":0.0'
    def pred_of(j):
        try: return float(j.get("probability_positive"))
        except: return None
else:  # multilabel
    PRED_SCHEMA='Output ONLY JSON: {"endpoints": {"<endpoint_id>": <prob 0..1>, ... one entry per endpoint}, "justification":"...", "unresolved_questions":[]}'
    REV_SCHEMA_TAIL='"endpoints": {"<endpoint_id>": <prob>, ...changed endpoints...}'
    def pred_of(j):
        e=j.get("endpoints")
        return e if isinstance(e,dict) else None

def base_prompt(target,refs):
    b=(f"TASK: {CFG['taskdef']}\nDESCRIPTOR GLOSSARY: {GLOSS}\n\n{fact_block(target)}\n\n"
       f"FIVE SIMILAR TRAINING REFERENCES (measured outcomes; these are labeled examples, NOT the target):\n{refs_block(refs)}\n\n"
       f"EXPLICIT TARGET-REFERENCE DIFFERENCES (descriptor + functional-group; NOT verified-structural):\n{diff_block(target,refs)}\n")
    if TT=="multilabel":
        b+=f"\nENDPOINTS (predict a probability for EACH): {', '.join(ENDPOINTS)}\n"
    return b

FROZEN_A={}   # idx -> parsed A dict; if set, run_mol reuses it instead of regenerating A
SKIP_B=False  # if True, skip the B review loop (not needed when only C-vs-A is studied)
def run_mol(target):
    f=PARTS/f"{SPLIT}_{target['idx']:04d}.json"
    if f.exists(): return
    base=base_prompt(target,refs_for(target))
    rec={"idx":target["idx"],"ds":DS,"model":MODEL,"split":SPLIT,"task_type":TT,
         "canonical":target["canonical"],"target_SEALED":(target.get("value") if TT=="regression" else target.get("label") if TT=="classification" else target.get("endpoints")),
         "A":None,"B":[],"C":{"rounds":[]}}
    # ---- A ----
    if target["idx"] in FROZEN_A:
        A=FROZEN_A[target["idx"]]; a_raw="[frozen-A]"; rec["A"]={"raw":a_raw,"parsed":A,"frozen":True}
    else:
        a_raw,_=bcall(base+"\nGive a concise assessment and your prediction.\n"+PRED_SCHEMA)
        A=xj(a_raw) or {}; rec["A"]={"raw":a_raw,"parsed":A}
    a_line=f"\nYOUR INITIAL PREDICTION: {json.dumps({k:A.get(k) for k in ('prediction_value','prediction','probability_positive','endpoints') if k in A})[:400]}\n"
    # ---- B: up to 3 reviews, no new evidence ----
    cur=A
    for r in (range(3) if not SKIP_B else []):
        bp=(base+f"\nYOUR CURRENT PREDICTION: {json.dumps({k:cur.get(k) for k in ('prediction_value','prediction','probability_positive','endpoints') if k in cur})[:400]}\n"
            "Critically REVIEW your reasoning and revise if warranted. You have NO new evidence beyond the above. "
            "If your prediction is already well-justified and stable, say so and STOP.\n"
            'Output ONLY JSON: {'+REV_SCHEMA_TAIL+',"changed":true,"stop":false,"justification":"..."}')
        br,_=bcall(bp); BJ=xj(br) or {}; rec["B"].append({"raw":br,"parsed":BJ})
        if any(k in BJ for k in ("prediction_value","prediction","probability_positive","endpoints")): cur=BJ
        if BJ.get("stop") or BJ.get("changed") is False: break
    rec["B_final"]=cur
    # ---- C: up to 3 evidence rounds ----
    if BUNDLE_DATA is None:
        rec["C"]["disabled"]="bundle_not_ready"; json.dump(rec,open(f,'w')); return
    ents,comp=available_entities(target)
    ent_menu=("AVAILABLE ENTITIES to query (select by entity_id):\n"
        f"  TASK entities: {[e['entity_id'] for e in ents if e['entity_type']=='task']}\n"
        f"  COMPOUND entity for THIS target: {('%s (has indexed findings; attributes present: %s)'%(comp['entity_id'],comp['attributes_present'][:25])) if comp else 'NONE (this molecule has no indexed literature findings)'}\n"
        f"AVAILABLE TASK ATTRIBUTES: {BUNDLE_DATA['entities']['task_attributes'][:40]}\n"
        f"AVAILABLE COMPOUND ATTRIBUTES: {BUNDLE_DATA['entities']['compound_attributes'][:40]}\n")
    searches=0; carried=""; cur=A
    for rnd in range(3):
        qp=(base+a_line+(f"\nEVIDENCE SO FAR:\n{carried}\n" if carried else "")+"\n"+ent_menu+"\n"+TOOL_CATALOG+"\n"
            +("For SIDER, make the question ENDPOINT-SPECIFIC (name the organ-class endpoint) and select that endpoint's task entity.\n" if TT=="multilabel" else "")+
            "Identify the MOST IMPORTANT unresolved, prediction-relevant question, then SELECT which entities and attributes to retrieve for it. "
            "CONTINUE ONLY IF a specific unresolved question exists AND literature retrieval over the selected entities/attributes could change the prediction. Otherwise STOP. Do not continue merely because you are uncertain.\n"
            'Output ONLY JSON: {"stop":false,"question":"...","entities":["<entity_id>",...],"attributes":["<attribute_id>",...],"route":"literature|structural|none","why_matters":"..."'
            +(',"endpoint":"<endpoint_id>"' if TT=="multilabel" else '')+'}')
        q_raw,_=bcall(qp); Q=xj(q_raw) or {}
        route=Q.get("route","literature")
        if Q.get("stop") or not Q.get("question") or route=="none":
            rec["C"]["rounds"].append({"ask":{"raw":q_raw,"parsed":Q},"stopped":True,"stop_reason":"no unresolved question with a usable retrieval route"}); break
        if route=="structural":
            ev={"note":"structural route requested but NO structural tool is implemented for this dataset; question UNRESOLVED"}; evstat="unresolved_no_tool"; passages=[]
        elif searches<3:
            sel_e=[e for e in (Q.get("entities") or []) if isinstance(e,str)] or [e["entity_id"] for e in ents if e["entity_type"]=="task"][:1]
            sel_a=[a for a in (Q.get("attributes") or []) if isinstance(a,str)]
            passages,rs=do_retrieve(Q["question"],sel_e,sel_a,k=3); searches+=1
            ev={"selected_entities":sel_e,"selected_attributes":sel_a,"passages":passages,"retrieval_status":rs}; evstat="retrieval"
        else:
            ev={"note":"retrieval budget exhausted"}; evstat="budget"; passages=[]
        evtxt=json.dumps(ev)[:3000]
        ap=(base+a_line+f"\nQUESTION: {Q['question']}\nEVIDENCE RETURNED ({evstat}):\n{evtxt}\n"
            "Assess whether the retrieved passages SUPPORT a claim and APPLY to THIS target. An unanswered/irrelevant search is NOT evidence; a value for a different compound does not transfer. Update your prediction ONLY if warranted.\n"
            'Output ONLY JSON: {'+REV_SCHEMA_TAIL+',"question_answer_status":"answered|partially_answered|not_answered|unresolved","evidence_subject":"target|reference|related_compound|general_class|unresolved","new_information_added":true,"next_action":"continue|stop","justification":"..."}')
        as_raw,_=bcall(ap); AS=xj(as_raw) or {}
        if any(k in AS for k in ("prediction_value","prediction","probability_positive","endpoints")): cur=AS
        carried+=f"\n[round{rnd+1}] Q: {Q['question'][:110]} -> {AS.get('question_answer_status')}, subj={AS.get('evidence_subject')}"
        rec["C"]["rounds"].append({"ask":{"raw":q_raw,"parsed":Q},"evidence":ev,"evidence_type":evstat,"assess":{"raw":as_raw,"parsed":AS}})
        if AS.get("next_action")=="stop": break
    # final C = last round with a parsed prediction, else A
    lastC=None
    for r in reversed(rec["C"]["rounds"]):
        pj=r.get("assess",{}).get("parsed")
        if pj and any(k in pj for k in ("prediction_value","prediction","probability_positive","endpoints")): lastC=pj; break
    rec["C"]["final"]=lastC or A
    rec["calls_so_far"]=_calls[0]
    json.dump(rec,open(f,'w'))

if __name__=="__main__":
    rows=load_molecules(SPLIT)
    todo=[r for r in rows if not (PARTS/f"{SPLIT}_{r['idx']:04d}.json").exists()]
    if NSHARD>1: todo=[r for r in todo if r['idx']%NSHARD==SHARDIDX]
    print(f"[{DS}/{MODEL}] {SPLIT}: {len(todo)} todo (shard {SHARDIDX}/{NSHARD}) | task_type={TT} | bundle={'ready' if BUNDLE_DATA else 'MISSING'} | endpoints={len(ENDPOINTS)}",flush=True)
    # strip mol objects are not picklable across processes; use threads
    for r in rows: r.pop("mol",None) if False else None
    def _safe(r):
        try: return run_mol(r)
        except RuntimeError as e: print(f"[skip idx={r.get('idx')}: {str(e)[:120]}]",flush=True)
    with ThreadPoolExecutor(max_workers=W) as ex: list(ex.map(_safe,todo))
    print(f"[{DS}/{MODEL}] DONE {SPLIT}; calls {_calls[0]}",flush=True)
    try: open(PARTS/f".done_{NSHARD}_{SHARDIDX}","w").write("done")
    except Exception: pass
