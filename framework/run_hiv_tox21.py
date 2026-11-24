"""Self-healing sharded watchdog: run condition-C framework for the 4 API models on hiv (4112, sharded)
+ tox21 (783, single). Resumable (part files + per-shard .done markers). Relaunches dead shards
(login node kills nohup). Writes HIV_TOX21_DONE.flag when every shard has its .done marker."""
import glob,time,subprocess,os,sys
from pathlib import Path
_HERE=Path(__file__).resolve().parent; _REPO=_HERE.parent
CODE=_HERE                                                        # directory holding run_fw_multi.py
DATA=Path(os.environ.get('MOLERAG_CARDS', _REPO/'data'/'cards'))  # corpus bundles (meta.json per dataset)
RUNS=Path(os.environ.get('MOLERAG_RUNS',  _REPO/'runs'))
LOGS=_REPO/'logs'; LOGS.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True)
PY=sys.executable
MODELS=["qwen3_32b","ministral_3_14b","opus_4_6","gpt5mini"]
CFG={"qwen3_32b":(6,3),"ministral_3_14b":(6,3),"opus_4_6":(3,2),"gpt5mini":(4,2)}  # (W, hiv_shards); tox21 uses 1 shard
def plines():
    o=subprocess.run(['pgrep','-af','run_fw_multi.py'],capture_output=True,text=True).stdout
    return [l for l in o.splitlines() if 'run_fw_multi.py' in l and 'sh -c' not in l and '/bin/sh' not in l]
def launch(ds,m,w,ns,k):
    cmd=(f'cd {CODE} && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
         f'nohup {PY} run_fw_multi.py {ds} {m} {w} {ns} {k} > {LOGS}/{ds}__{m}_s{k}of{ns}.log 2>&1 </dev/null &')
    subprocess.Popen(cmd,shell=True)
def shards():
    for ds in ["tox21","hiv"]:
        if not (DATA/ds/'meta.json').exists(): continue
        for m in MODELS:
            w,hs=CFG[m]; ns=(hs if ds=="hiv" else 1)
            for k in range(ns): yield ds,m,w,ns,k
while True:
    ls=plines(); pending=0
    for ds,m,w,ns,k in shards():
        if (RUNS/f'{ds}__{m}'/f'.done_{ns}_{k}').exists(): continue
        pending+=1
        sig=f'run_fw_multi.py {ds} {m} {w} {ns} {k}'
        if not any(sig in l for l in ls): launch(ds,m,w,ns,k); time.sleep(4)
    st=" ".join(f"{ds[:3]}/{m[:3]}={len(glob.glob(str(RUNS/(ds+'__'+m)/'test_*.json')))}" for ds in ['tox21','hiv'] if (DATA/ds/'meta.json').exists() for m in MODELS)
    print(f"[{time.strftime('%H:%M:%S')}] pending_shards={pending} | {st}",flush=True)
    if pending==0: break
    time.sleep(50)
open(RUNS/'HIV_TOX21_DONE.flag','w').write('done'); print("ALL HIV+TOX21 SHARDS DONE",flush=True)
