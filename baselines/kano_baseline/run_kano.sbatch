#!/bin/bash
#SBATCH --job-name=kano_cmpnn
#SBATCH --partition=gpuA100x4
#SBATCH --account=bhrm-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=20:00:00
#SBATCH --output=/workspace/mole-rag_naacl/kano_baseline/execution/kano_%x_%j.log

set -u
ENC=${ENC:-CMPNN}
CKPT=${CKPT:-./dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl}
TAG=${TAG:-cmpnn}
NRUNS=${NRUNS:-1}

KANO=/workspace/mole-rag_naacl/kano/KANO
DATA=/workspace/mole-rag_naacl/kano_baseline/data
RES=/workspace/mole-rag_naacl/kano_baseline/results_${TAG}
PY=/workspace/miniconda3/envs/kano/bin/python
mkdir -p "$RES"
cd "$KANO"
echo "host $(hostname)  ENC=$ENC  TAG=$TAG  NRUNS=$NRUNS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# small -> large so bbbp is an early smoke test and hiv is last
declare -A DTYPE=( [bbbp]=classification [bace]=classification [clintox]=classification [sider]=classification [tox21]=classification [hiv]=classification [esol]=regression [freesolv]=regression [lipo]=regression )
declare -A METRIC=( [bbbp]=auc [bace]=auc [clintox]=auc [sider]=auc [tox21]=auc [hiv]=auc [esol]=rmse [freesolv]=rmse [lipo]=rmse )
for t in bbbp bace clintox esol freesolv lipo sider tox21 hiv; do
  echo "===== KANO-$TAG $t ($(date +%H:%M:%S)) ====="
  CKPT_ARG=""
  if [ "$CKPT" != "NONE" ]; then CKPT_ARG="--checkpoint_path $CKPT"; fi
  $PY train.py \
    --data_path "$DATA/$t/train.csv" \
    --separate_val_path "$DATA/$t/val.csv" \
    --separate_test_path "$DATA/$t/test.csv" \
    --dataset_type "${DTYPE[$t]}" \
    --metric "${METRIC[$t]}" \
    --epochs 100 --num_runs $NRUNS --gpu 0 --batch_size 256 --seed 51 --init_lr 1e-4 \
    --split_type scaffold_balanced \
    --step functional_prompt \
    --encoder_name "$ENC" \
    --exp_name "kano_${TAG}" --exp_id "$t" \
    $CKPT_ARG \
    > "$RES/${t}.log" 2>&1
  echo "  -> $(grep -aE 'Overall test|^Results:' "$RES/${t}.log" | tail -2 | tr '\n' ' ')"
done
echo "KANO_${TAG}_ALL_DONE"
