#!/bin/bash
# GPU run: YAIB Transformer on miiv/mortality24, 1 rep x 1 fold, PAPER's chosen HPs.
# Hyperparameters from Vandewater et al. (YAIB, ICLR 2024), Table 27 "Chosen hyperparameters
# for transformer", row Mortality / MIMIC-IV, plus fixed defaults from Table 22:
#   Learning Rate 6.18e-05 | Dropout 0.165 | Heads 1 | Hidden 48 | Depth 3
#   ff_hidden_mult 2 | l1_reg 0.0 | weight_decay 1e-6 (gin default) | batch 64 | epochs<=1000 (patience 10)
# Attention dropout (dropout_att) is not separately reported in Table 27 -> left at model default 0.0.
# No --tune (all HPs are fixed scalars -> deterministic), no --cpu (GPU), no -wd (no W&B).
#SBATCH --job-name=yaib_gpu_tf_php
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/gpu_transformer_paperhp_%j.log
#SBATCH --error=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/gpu_transformer_paperhp_%j.log

set -uo pipefail
ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
COH=$ROOT/ehrapy-usecase/processed_data/yaib-cohorts
LOG=$ROOT/ehrapy-usecase/processed_data/yaib-benchmark
IMG=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images/nvidia+pytorch+26.03-py3_yaib.sqsh
mkdir -p /localscratch/$USER/enroot/tmp "$LOG/_tmp"

echo "==== NODE $(hostname) | job $SLURM_JOB_ID | $(date) ===="
nvidia-smi -L || echo "WARN: nvidia-smi (host) failed"

enroot list 2>/dev/null | grep -qx yaib && enroot remove -f yaib || true
enroot create --name yaib "$IMG"

# TMPDIR=/tmp is REQUIRED: DL DataLoader workers share GPU tensors over an AF_UNIX socket under
# TMPDIR; a long shared-tree path overflows the 108-char socket limit. See smoke_gpu_transformer.sbatch.
enroot start --root --mount "$ROOT":"$ROOT" --env TMPDIR=/tmp yaib bash -c '
  source /opt/yaib-venv/bin/activate
  python -c "import torch; print(\"torch\", torch.__version__, \"| cuda\", torch.cuda.is_available(), \"|\", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"))"
  cd "'"$YAIB"'"
  icu-benchmarks train \
    -d "'"$COH"'/mortality24/miiv" -n miiv \
    -t BinaryClassification -tn Mortality24 -m Transformer \
    -l "'"$LOG"'" -s 1111 -gc \
    -hp \
      execute_repeated_cv.cv_repetitions_to_train=1 \
      execute_repeated_cv.cv_folds_to_train=1 \
      optimizer/hyperparameter.lr=6.18e-05 \
      model/hyperparameter.hidden=48 \
      model/hyperparameter.heads=1 \
      model/hyperparameter.depth=3 \
      model/hyperparameter.dropout=0.165 \
      model/hyperparameter.dropout_att=0.0 \
      model/hyperparameter.ff_hidden_mult=2 \
      model/hyperparameter.l1_reg=0.0 \
      train_common.epochs=1000
'
RC=$?
echo "==== icu-benchmarks exit=$RC | $(date) ===="
enroot remove -f yaib || true
exit $RC
