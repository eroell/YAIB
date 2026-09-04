#!/bin/bash
# YAIB DL matrix on miiv (MIMIC-IV): one model x 5 tasks, 1 rep x 1 fold each, GPU, paper HPs.
# Parametrized by $MODEL (Transformer|GRU|LSTM|TCN); submit once per model:
#   sbatch -J dl_Transformer --export=ALL,MODEL=Transformer run_dl_matrix.sbatch   (etc.)
# Paper HPs: Vandewater et al. (YAIB, ICLR 2024) Tables 24(GRU) 25(LSTM) 26(TCN) 27(Transformer),
# MIMIC-IV rows; fixed defaults from Table 22 (epochs<=1000/patience 10/batch 64/weight_decay 1e-6).
# No --tune (fixed scalars -> deterministic), no --cpu (GPU), no -wd (no W&B).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/dlmatrix_%x_%j.log
#SBATCH --error=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/dlmatrix_%x_%j.log

set -uo pipefail
: "${MODEL:?set MODEL=Transformer|GRU|LSTM|TCN via --export=ALL,MODEL=...}"
ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
COH=$ROOT/ehrapy-usecase/processed_data/yaib-cohorts
LOG=$ROOT/ehrapy-usecase/processed_data/yaib-benchmark
IMG=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images/nvidia+pytorch+26.03-py3_yaib.sqsh
SUMMARY=$LOG/_logs/dlmatrix_${MODEL}_summary.tsv
# Unique container name per JOB: enroot's data path is per-NODE (not per-job), so two jobs on the
# same node must NOT share a container name (they'd clobber each other's rootfs -> both fail fast).
CNAME=yaib_${SLURM_JOB_ID:-$$}
mkdir -p /localscratch/$USER/enroot/tmp "$LOG/_logs" "$LOG/_tmp"
echo "==== NODE $(hostname) | job $SLURM_JOB_ID | MODEL=$MODEL | CNAME=$CNAME | $(date) ===="; nvidia-smi -L || true

enroot list 2>/dev/null | grep -qx "$CNAME" && enroot remove -f "$CNAME" || true
enroot create --name "$CNAME" "$IMG"

# task -> gin task type + experiment label
# los uses RegressionLoS (outcome 0-168h); Regression.gin's 0-15 is only correct for kidney_function.
declare -A TTYPE=( [mortality24]=BinaryClassification [aki]=BinaryClassification [sepsis]=BinaryClassification \
                   [kidney_function]=Regression [los]=RegressionLoS )
declare -A TN=( [mortality24]=Mortality24 [aki]=Aki [sepsis]=Sepsis [kidney_function]=KidneyFunction [los]=LengthOfStay )

# model-specific per-task HP bindings (MIMIC-IV rows) + model-common fixed bindings
declare -A HP
case "$MODEL" in
  Transformer)
    MCOMMON="model/hyperparameter.ff_hidden_mult=2 model/hyperparameter.l1_reg=0.0 model/hyperparameter.dropout_att=0.0"
    HP[mortality24]="optimizer/hyperparameter.lr=6.18e-05 model/hyperparameter.hidden=48  model/hyperparameter.heads=1 model/hyperparameter.depth=3 model/hyperparameter.dropout=0.165"
    HP[aki]="optimizer/hyperparameter.lr=1.22e-04 model/hyperparameter.hidden=72  model/hyperparameter.heads=2 model/hyperparameter.depth=3 model/hyperparameter.dropout=0.0413"
    HP[sepsis]="optimizer/hyperparameter.lr=3.53e-05 model/hyperparameter.hidden=98  model/hyperparameter.heads=1 model/hyperparameter.depth=3 model/hyperparameter.dropout=0.367"
    HP[kidney_function]="optimizer/hyperparameter.lr=1.53e-04 model/hyperparameter.hidden=160 model/hyperparameter.heads=5 model/hyperparameter.depth=2 model/hyperparameter.dropout=0.0862"
    HP[los]="optimizer/hyperparameter.lr=1.13e-04 model/hyperparameter.hidden=76  model/hyperparameter.heads=3 model/hyperparameter.depth=2 model/hyperparameter.dropout=0.0411"
    ;;
  GRU)
    MCOMMON=""
    HP[mortality24]="optimizer/hyperparameter.lr=1.43e-04 model/hyperparameter.hidden_dim=77  model/hyperparameter.layer_dim=2"
    HP[aki]="optimizer/hyperparameter.lr=2.82e-04 model/hyperparameter.hidden_dim=139 model/hyperparameter.layer_dim=3"
    HP[sepsis]="optimizer/hyperparameter.lr=2.39e-04 model/hyperparameter.hidden_dim=52  model/hyperparameter.layer_dim=3"
    HP[kidney_function]="optimizer/hyperparameter.lr=1.11e-04 model/hyperparameter.hidden_dim=148 model/hyperparameter.layer_dim=1"
    HP[los]="optimizer/hyperparameter.lr=1.57e-05 model/hyperparameter.hidden_dim=237 model/hyperparameter.layer_dim=2"
    ;;
  LSTM)
    MCOMMON=""
    HP[mortality24]="optimizer/hyperparameter.lr=3.00e-04 model/hyperparameter.hidden_dim=185 model/hyperparameter.layer_dim=2"
    HP[aki]="optimizer/hyperparameter.lr=3.00e-04 model/hyperparameter.hidden_dim=256 model/hyperparameter.layer_dim=3"
    HP[sepsis]="optimizer/hyperparameter.lr=2.46e-04 model/hyperparameter.hidden_dim=161 model/hyperparameter.layer_dim=1"
    HP[kidney_function]="optimizer/hyperparameter.lr=2.49e-04 model/hyperparameter.hidden_dim=256 model/hyperparameter.layer_dim=1"
    HP[los]="optimizer/hyperparameter.lr=1.80e-05 model/hyperparameter.hidden_dim=253 model/hyperparameter.layer_dim=3"
    ;;
  TCN)
    MCOMMON=""
    HP[mortality24]="optimizer/hyperparameter.lr=1.81e-05 model/hyperparameter.num_channels=130 model/hyperparameter.kernel_size=3  model/hyperparameter.dropout=0.350"
    HP[aki]="optimizer/hyperparameter.lr=2.98e-04 model/hyperparameter.num_channels=249 model/hyperparameter.kernel_size=3  model/hyperparameter.dropout=0.0761"
    HP[sepsis]="optimizer/hyperparameter.lr=2.23e-04 model/hyperparameter.num_channels=78  model/hyperparameter.kernel_size=4  model/hyperparameter.dropout=0.106"
    HP[kidney_function]="optimizer/hyperparameter.lr=1.88e-05 model/hyperparameter.num_channels=33  model/hyperparameter.kernel_size=2  model/hyperparameter.dropout=0.215"
    HP[los]="optimizer/hyperparameter.lr=1.57e-04 model/hyperparameter.num_channels=44  model/hyperparameter.kernel_size=12 model/hyperparameter.dropout=0.167"
    ;;
  *) echo "unknown MODEL=$MODEL"; exit 2;;
esac

# Run all 5 tasks by default; set ONLYTASK=<task> to run just one (e.g. slow TCN, parallelized per task).
TASKLIST=(mortality24 aki sepsis kidney_function los)
[ -n "${TASKS:-}" ] && TASKLIST=($TASKS)          # space-separated subset, e.g. "kidney_function los"
[ -n "${ONLYTASK:-}" ] && TASKLIST=("$ONLYTASK")
for TASK in "${TASKLIST[@]}"; do
  RUNLOG=$LOG/_logs/dlmatrix_${MODEL}_${TASK}.log
  echo "##### RUN model=$MODEL task=$TASK type=${TTYPE[$TASK]} | $(date +%H:%M:%S) -> $RUNLOG"
  enroot start --root --mount "$ROOT":"$ROOT" --env TMPDIR=/tmp "$CNAME" bash -c '
    source /opt/yaib-venv/bin/activate
    cd "'"$YAIB"'"
    icu-benchmarks train -d "'"$COH"'/'"$TASK"'/miiv" -n miiv \
      -t '"${TTYPE[$TASK]}"' -tn '"${TN[$TASK]}"' -m '"$MODEL"' -l "'"$LOG"'" -s 1111 -gc \
      -hp execute_repeated_cv.cv_repetitions_to_train=1 execute_repeated_cv.cv_folds_to_train=1 \
          train_common.epochs=1000 '"$MCOMMON"' '"${HP[$TASK]}"'
  ' > "$RUNLOG" 2>&1
  RC=$?
  RES=$(grep -a "Accumulated results" "$RUNLOG" | tail -1 | sed "s/.*Accumulated results: //")
  printf '%s\t%s\t%s\trc=%s\t%s\n' "$MODEL" "$TASK" "${TTYPE[$TASK]}" "$RC" "${RES:-<no result line>}" >> "$SUMMARY"
  echo "   -> rc=$RC ${RES:0:120}"
done

enroot remove -f "$CNAME" || true
echo "==== DONE MODEL=$MODEL | $(date) | summary: $SUMMARY ===="
