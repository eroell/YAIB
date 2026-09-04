#!/usr/bin/env bash
# =============================================================================
# YAIB baseline training — Phase 1 step (b). CPU (LGBM / LogReg / ElasticNet).
# Deep models on GPU: use run_dl_matrix.sbatch instead.
#
# Everything is driven by env vars, so one script covers the whole matrix:
#
#   sbatch run_yaib.sbatch                                    # miiv/mortality24/LGBM, 1x1 smoke
#   TASK=aki SRC=hirid            sbatch --export=ALL run_yaib.sbatch
#   TASK=los SRC=miiv MODEL=LGBMRegressor REPS=5 FOLDS=5 sbatch --export=ALL run_yaib.sbatch
#   COH=<other cohort dir>        sbatch --export=ALL run_yaib.sbatch   # e.g. the debug cohorts
#
#   # whole matrix, one job each (unique container name per job is handled below):
#   for S in miiv eicu hirid; do for T in mortality24 aki sepsis; do
#     TASK=$T SRC=$S sbatch --export=ALL -J yaib_${T}_${S} run_yaib.sbatch; done; done
#
# Cost: 1 rep x 1 fold ~6 min · 1x5 ~30 min · 5x5 ~2.5 h. The bottleneck is the per-fold
# "caching dataset in ram" (~5 min), NOT the model fit (~8 s for LGBM) — so wall time scales
# with FOLDS, not with model complexity.
#
# Compliance: no -wd (W&B would push cohort-derived data off-site). All paths stay in $ROOT.
# =============================================================================
#SBATCH --job-name=yaib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/sbatch_%x_%j.log
#SBATCH --error=/ictstr01/groups/shared/physionet-credentialized/ehrapy-usecase/processed_data/yaib-benchmark/_logs/sbatch_%x_%j.log

set -uo pipefail

ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
COH="${COH:-$ROOT/ehrapy-usecase/processed_data/yaib-cohorts}"
LOG="${LOG:-$ROOT/ehrapy-usecase/processed_data/yaib-benchmark}"
IMG=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images/nvidia+pytorch+26.03-py3_yaib.sqsh

TASK="${TASK:-mortality24}"
SRC="${SRC:-miiv}"
REPS="${REPS:-1}"
FOLDS="${FOLDS:-1}"
SEED="${SEED:-1111}"

# task -> (-t TaskType gin, -tn free-form label, default model).
# NOTE los uses RegressionLoS: plain Regression.gin hardcodes outcome_max=15 (kidney function,
# mg/dL), which mis-scales LoS. Denormalize MAE with x15 (KF, mg/dL) / x168 (LoS, hours).
case "$TASK" in
  mortality24)      TTYPE=BinaryClassification; LABEL=Mortality24;    DEFMODEL=LGBMClassifier ;;
  aki)              TTYPE=BinaryClassification; LABEL=Aki;            DEFMODEL=LGBMClassifier ;;
  sepsis)           TTYPE=BinaryClassification; LABEL=Sepsis;         DEFMODEL=LGBMClassifier ;;
  kidney_function)  TTYPE=Regression;           LABEL=KidneyFunction; DEFMODEL=LGBMRegressor  ;;
  los)              TTYPE=RegressionLoS;        LABEL=LengthOfStay;   DEFMODEL=LGBMRegressor  ;;
  *) echo "Unknown TASK='$TASK' (mortality24|aki|sepsis|kidney_function|los)" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DEFMODEL}"

DATA="$COH/$TASK/$SRC"
[ -d "$DATA" ] || { echo "No cohort at $DATA" >&2; exit 2; }

mkdir -p "$LOG/_logs" "$LOG/_tmp"

echo "==== node $(hostname) | job ${SLURM_JOB_ID:-<none>} | $(date) ===="
echo "data   : $DATA"
echo "task   : $TASK  -t $TTYPE  -tn $LABEL"
echo "model  : $MODEL | ${REPS} rep(s) x ${FOLDS} fold(s) | seed $SEED"
echo "logdir : $LOG"
echo

# enroot's ENROOT_DATA_PATH is per-NODE, not per-job: two jobs landing on the same node with
# the same container name clobber each other's rootfs and both die in ~40 s. Unique name per job.
CNAME="yaib_${SLURM_JOB_ID:-manual$$}"
cleanup() { enroot remove -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

enroot list 2>/dev/null | grep -qx "$CNAME" && enroot remove -f "$CNAME"
enroot create --name "$CNAME" "$IMG"

# CPU node: NVIDIA_VISIBLE_DEVICES=void skips the enroot nvidia hook (harmless banner otherwise).
enroot start --mount "$ROOT":"$ROOT" \
  --env NVIDIA_VISIBLE_DEVICES=void --env TMPDIR="$LOG/_tmp" --env MPLCONFIGDIR="$LOG/_tmp/mpl" \
  "$CNAME" bash -c "
    source /opt/yaib-venv/bin/activate && cd '$YAIB' && \
    icu-benchmarks train \
      -d '$DATA' -n '$SRC' \
      -t $TTYPE -tn $LABEL -m '$MODEL' \
      -l '$LOG' -s $SEED --cpu -gc \
      -hp execute_repeated_cv.cv_repetitions_to_train=$REPS \
          execute_repeated_cv.cv_folds_to_train=$FOLDS
  "
RC=$?

echo
echo "==== icu-benchmarks exit=$RC | $(date) ===="
echo "Headline metrics: grep -E 'Accumulated results' on this log."
exit $RC
