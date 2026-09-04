#!/bin/bash
# GPU smoke test: YAIB Transformer on miiv/mortality24, 1 rep x 1 fold, no tuning.
# Same "manner" as the LGBM CPU smoke test, but on a GPU and with -m Transformer.
#SBATCH --job-name=yaib_gpu_tf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/gpu_transformer_%j.log
#SBATCH --error=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/gpu_transformer_%j.log

set -uo pipefail
ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
COH=$ROOT/ehrapy-usecase/processed_data/yaib-cohorts
LOG=$ROOT/ehrapy-usecase/processed_data/yaib-benchmark
IMG=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images/nvidia+pytorch+26.03-py3_yaib.sqsh
mkdir -p /localscratch/$USER/enroot/tmp "$LOG/_tmp"

echo "==== NODE $(hostname) | job $SLURM_JOB_ID | $(date) ===="
nvidia-smi -L || echo "WARN: nvidia-smi (host) failed"

# recreate ephemeral container on THIS gpu node (image default NVIDIA_VISIBLE_DEVICES=all -> GPU exposed)
enroot list 2>/dev/null | grep -qx yaib && enroot remove -f yaib || true
enroot create --name yaib "$IMG"

# GPU visibility check inside the container, then run the Transformer (NO --cpu, NO -wd)
# TMPDIR MUST be short: DataLoader workers share GPU tensors via an AF_UNIX socket under
# TMPDIR, and the socket path has a 108-char limit. The long shared-tree path overflows it
# ("OSError: AF_UNIX path too long"). Container-internal /tmp is short and backed by
# node-local localscratch (~590G) -> safe on both path length and space. Model outputs still
# go to -l (shared tree); only the ephemeral IPC socket lives here.
enroot start --root --mount "$ROOT":"$ROOT" --env TMPDIR=/tmp yaib bash -c '
  source /opt/yaib-venv/bin/activate
  echo "---- torch CUDA check ----"
  python -c "import torch; print(\"torch\", torch.__version__, \"| cuda_available\", torch.cuda.is_available(), \"| n_gpu\", torch.cuda.device_count(), \"|\", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"))"
  echo "---- launching Transformer (1 rep x 1 fold, no tune) ----"
  cd "'"$YAIB"'"
  icu-benchmarks train \
    -d "'"$COH"'/mortality24/miiv" -n miiv \
    -t BinaryClassification -tn Mortality24 -m Transformer \
    -l "'"$LOG"'" -s 1111 -gc \
    -hp execute_repeated_cv.cv_repetitions_to_train=1 execute_repeated_cv.cv_folds_to_train=1
'
RC=$?
echo "==== icu-benchmarks exit=$RC | $(date) ===="
enroot remove -f yaib || true
exit $RC
