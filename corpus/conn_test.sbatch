#!/bin/bash
#SBATCH --job-name=e70conn
#SBATCH --account=bhrm-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=${EXP70_BASE:-$(cd "$(dirname "$0")/.." && pwd)/corpus_runs}/logs/conn_%j.out
PY=${PY:-python3}
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID $(date -u)"
export AWS_BEARER_TOKEN_BEDROCK=$(grep -o 'AWS_BEARER_TOKEN_BEDROCK=.*' ${BEDROCK_ENV:-$HOME/.config/molrag/bedrock.env} | cut -d= -f2-)
timeout 30 $PY - <<'PYEOF'
import socket, time
try:
    import requests
    r=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi',timeout=15)
    print('NCBI', r.status_code)
except Exception as e: print('NCBI FAIL', repr(e))
PYEOF
echo "--- S3 ---"; timeout 30 aws s3 ls s3://molerag/ >/dev/null 2>&1 && echo "S3 OK" || echo "S3 FAIL"
echo "--- Bedrock ---"
cd ${CODE:-$(cd "$(dirname "$0")" && pwd)}
EXP70_DATASET=lipo EXP70_CLAUDE_MODELS=us.anthropic.claude-haiku-4-5-20251001-v1:0 timeout 40 $PY -c "
import bedrock_llm as B
import time; t=time.time()
ok,txt,u,m=B.converse('Reply {\"ok\":true} only.','ping',max_tokens=30)
print('BEDROCK', ok, '%.2fs'%(time.time()-t), m.get('model') if m else None, repr(txt)[:60])
" 2>&1 | tail -3
echo "CONN_TEST_DONE"
