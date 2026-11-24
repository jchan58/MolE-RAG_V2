"""Shared structural checker for the aminoquinoline C-3 applicability check.
Enriched over the 2-molecule loop with an honest status:
  - 'no_core'    : confidently no 2-aminoquinoline core -> finding (about 2-aminoquinolines) not applicable
  - 'resolved'   : core found AND C2/C3 position pinned -> C-3 substituent present/absent is trustworthy
  - 'unresolved' : core SMARTS matched but C2/C3 could not be pinned (ambiguous) -> do not assert C-3 state
Feedback text is GENERATED from the computed fields (single code path, no molecule-id branching)."""
from rdkit import Chem
# 2-AMINOQUINOLINE = amino on the carbon adjacent to the ring-N of a FUSED bicyclic (quinoline) system.
# The amino carbon is match atom index 1. A bare 2-aminoPYRIDINE (monocyclic) does NOT match -> no_core.
_CORE=Chem.MolFromSmarts('[NX3;!$(N=*)]-c1ccc2ccccc2n1')
def c3_analysis(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return {"status":"unresolved","reason":"unparseable SMILES","smiles":smi}
    matches=m.GetSubstructMatches(_CORE)
    if not matches:
        return {"status":"no_core","aminoquinoline_core_present":False,"heavy_atoms":m.GetNumHeavyAtoms(),"smiles":smi}
    c2=m.GetAtomWithIdx(matches[0][1])  # the 2-amino carbon of the quinoline, pinned by the SMARTS
    # C3 = aromatic-C ring neighbour of C2 (NOT the ring N); quinoline C2 has exactly one such neighbour
    c3=[c for c in c2.GetNeighbors() if c.GetSymbol()=='C' and c.GetIsAromatic() and m.GetBondBetweenAtoms(c2.GetIdx(),c.GetIdx()).IsInRing()]
    if not c3:
        return {"status":"unresolved","aminoquinoline_core_present":True,
                "reason":"quinoline core matched but C-3 ring neighbour of C2 could not be identified",
                "heavy_atoms":m.GetNumHeavyAtoms(),"smiles":smi}
    c3_sub=None
    for c in c3:
        for nb in c.GetNeighbors():
            if nb.GetIdx()==c2.GetIdx(): continue
            b=m.GetBondBetweenAtoms(c.GetIdx(),nb.GetIdx())
            if (not b.IsInRing()) and nb.GetSymbol()!='H':
                seen=set([c.GetIdx()]); stack=[nb.GetIdx()]; frag=set()
                while stack:
                    x=stack.pop()
                    if x in seen: continue
                    seen.add(x); frag.add(x)
                    for y in m.GetAtomWithIdx(x).GetNeighbors():
                        if y.GetIdx() not in seen: stack.append(y.GetIdx())
                try: c3_sub=Chem.MolFragmentToSmiles(m,atomsToUse=sorted(frag))
                except Exception: c3_sub='(fragment)'
                break
        if c3_sub: break
    # substituent elsewhere on the FUSED aromatic ring system of C2 (distinguish 'bare' from 'substituted at another position')
    sys=set(); stack=[c2.GetIdx()]
    while stack:
        i=stack.pop()
        if i in sys: continue
        sys.add(i)
        for nb in m.GetAtomWithIdx(i).GetNeighbors():
            b=m.GetBondBetweenAtoms(i,nb.GetIdx())
            if nb.GetIsAromatic() and b.GetIsAromatic() and nb.GetIdx() not in sys: stack.append(nb.GetIdx())
    amino_idx={n.GetIdx() for n in c2.GetNeighbors() if n.GetSymbol()=='N' and not n.GetIsAromatic()}
    c3_frag=set()  # recompute C-3 fragment atoms to exclude them from 'elsewhere'
    for c in c3:
        for nb in c.GetNeighbors():
            if nb.GetIdx()==c2.GetIdx(): continue
            b=m.GetBondBetweenAtoms(c.GetIdx(),nb.GetIdx())
            if (not b.IsInRing()) and nb.GetSymbol()!='H':
                seen=set([c.GetIdx()]); st=[nb.GetIdx()]
                while st:
                    x=st.pop()
                    if x in seen: continue
                    seen.add(x); c3_frag.add(x)
                    for y in m.GetAtomWithIdx(x).GetNeighbors():
                        if y.GetIdx() not in seen: st.append(y.GetIdx())
    elsewhere=False
    for i in sys:
        at=m.GetAtomWithIdx(i)
        for nb in at.GetNeighbors():
            if nb.GetIdx() in sys or nb.GetIdx() in amino_idx or nb.GetIdx() in c3_frag: continue
            b=m.GetBondBetweenAtoms(at.GetIdx(),nb.GetIdx())
            if (not b.IsInRing()) and nb.GetSymbol()!='H': elsewhere=True
    return {"status":"resolved","aminoquinoline_core_present":True,
            "c3_sidechain_present":bool(c3_sub),"c3_sidechain_smiles":c3_sub,
            "substituent_elsewhere_on_core":bool(elsewhere),
            "heavy_atoms":m.GetNumHeavyAtoms(),"smiles":smi}
def gen_feedback(a):
    """Feedback text generated purely from the analysis dict (identical path for every molecule)."""
    if a["status"]=="unresolved":
        return ("VERIFIED STRUCTURAL CHECK (RDKit): the 2-aminoquinoline scaffold/position could NOT be identified reliably "
                f"for this molecule ({a.get('reason','ambiguous scaffold')}). The C-3 substituent state is UNRESOLVED; do not "
                "assume the finding's C-3 feature is present or absent.")
    if a["status"]=="no_core":
        return ("VERIFIED STRUCTURAL CHECK (RDKit, computed from this molecule's structure):\n"
                "- 2-aminoquinoline core: ABSENT. This molecule does not contain the 2-aminoquinoline scaffold the finding is about.\n"
                f"- total heavy atoms: {a['heavy_atoms']}")
    core="PRESENT"
    if a["c3_sidechain_present"]:
        c3=(f"SUBSTITUTED — a side chain is attached at the C-3 position (ring carbon adjacent to the 2-amino carbon). "
            f"Side-chain fragment (SMILES): {a['c3_sidechain_smiles']}.")
    else:
        extra=" A substituent is present elsewhere on the aminoquinoline ring system, but NOT at the C-3 position." if a.get("substituent_elsewhere_on_core") else " No side chain is present at that position."
        c3=("UNSUBSTITUTED — there is NO side chain at the C-3 position (ring carbon adjacent to the 2-amino carbon)."+extra)
    return ("VERIFIED STRUCTURAL CHECK (RDKit, computed from this molecule's structure):\n"
            f"- 2-aminoquinoline core: {core}\n- C-3 position: {c3}\n- total heavy atoms: {a['heavy_atoms']}")
# ---- FROZEN prompt templates + passage (one source of truth for every condition) ----
PASSAGE=("integrate the C-3 side chain on 2-aminoquinoline extending deep into the P2' binding pocket of beta-secretase "
         "enhancing the ligand's potency")
SRC="PMC6034716 (about a 2-aminoquinoline inhibitor series / related molecules)"
def ask_prompt(mid,facts):
    return (f"You are evaluating molecule {mid}. Below are its molecular facts and ONE literature finding that is about "
       f"RELATED molecules (a chemical series), not necessarily this one.\nMOLECULAR FACTS:\n{facts}\n\nLITERATURE FINDING "
       f"(about related molecules): \"{PASSAGE}\" (source: {SRC})\n\nBefore using this finding for THIS molecule, answer:\n"
       "- requirements: what must be true about THIS molecule for the finding to apply to it?\n- established: what the finding "
       "establishes, and for which molecules.\n- needs_checking: specific things to verify about THIS molecule before relying on "
       "the finding (each as {check, why}).\nDo NOT give a BACE-1 prediction yet.\n"
       'Output ONLY JSON: {"requirements":["..."],"established":["..."],"needs_checking":[{"check":"...","why":"..."}]}')
def predict_prompt(mid,facts,model_question=None,feedback=None):
    """Baseline = feedback None; Checked = pass the model's requested question + computed feedback. Identical otherwise."""
    p=(f"You are predicting whether molecule {mid} is an ACTIVE or INACTIVE inhibitor of human BACE-1 (beta-secretase 1).\n"
       f"MOLECULAR FACTS:\n{facts}\n\nLITERATURE FINDING (about a related 2-aminoquinoline inhibitor series): "
       f"\"{PASSAGE}\" (source: {SRC})\n")
    if feedback is not None:
        p+=(f"\nTo decide whether this finding applies to {mid}, this structural check was requested:\n  \"{model_question}\"\n"
            f"A structural tool (RDKit, computed from this molecule's structure) answered:\n{feedback}\n")
    p+=("\nPredict this molecule's BACE-1 activity. You may answer \"active\", \"inactive\", or \"uncertain\".\n"
        "List the specific structural claims about THIS molecule you are relying on.\n"
        'Output ONLY JSON: {"prediction":"active|inactive|uncertain","probability_active":0.0,'
        '"structural_claims":["..."],"justification":"...","key_evidence":"..."}')
    return p
if __name__=="__main__":
    import json
    tests=[("feature_present_C3","Cc1cc2ccccc2nc1N"),
           ("feature_absent_bare","Nc1ccc2ccccc2n1"),
           ("substituent_different_position_C6","Cc1ccc2nc(N)ccc2c1"),
           ("aminopyridine_NOT_quinoline","Nc1ccccn1"),
           ("aminopyridine_3sub_NOT_quinoline","n1cccc(NCc2ccccc2)c1N"),
           ("no_scaffold_benzoic_acid","c1ccccc1C(=O)O"),
           ("no_scaffold_aliphatic","CCCCNC(=O)C"),
           ("real_A_m1352","Cc1ccccc1-c1ccc2nc(N)c(CCC(=O)NC3CCCCC3)cc2c1"),
           ("real_B_bare_m1454","Nc1ccc2ccccc2n1"),
           ("real_B_thiophene_offpos","s1cc(cc1)-c1cc2c(nc(N)cc2)cc1")]
    for name,smi in tests:
        a=c3_analysis(smi)
        print("="*80);print(name,smi)
        print("  ",{k:a.get(k) for k in ("status","aminoquinoline_core_present","c3_sidechain_present","c3_sidechain_smiles","substituent_elsewhere_on_core")})
        print("  FEEDBACK:",gen_feedback(a).replace("\n"," | "))
