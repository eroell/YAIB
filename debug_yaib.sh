#!/usr/bin/env bash
# =============================================================================
# Interactive / scripted Python entry point into the YAIB container.
#
#   bash debug_yaib.sh                  # bash shell inside the container, venv active, cwd=$YAIB
#   bash debug_yaib.sh python           # plain python REPL
#   bash debug_yaib.sh ipython          # ipython REPL if present, else python
#   bash debug_yaib.sh -c '<python>'    # run a python snippet
#   bash debug_yaib.sh -- <any cmd>     # run an arbitrary command in the container
#
#   GPU=1 bash debug_yaib.sh            # do NOT set NVIDIA_VISIBLE_DEVICES=void (needs a gpu_p job)
#   COH=<dir> LOG=<dir> bash debug_yaib.sh    # override cohort / output dirs
#
# WHY THIS WORKS FOR DEBUGGING: the image installs icu_benchmarks EDITABLE against this
# checkout (`import icu_benchmarks` resolves to $YAIB/icu_benchmarks), so anything you edit in
# the repo takes effect on the next run inside the container — no rebuild. That makes the
# normal workflow:
#
#   1. drop `breakpoint()` into e.g. icu_benchmarks/data/preprocessor.py
#   2. bash debug_yaib.sh          # then, at the container prompt:
#      icu-benchmarks train -d "$COH/mortality24/miiv" -n miiv \
#        -t BinaryClassification -tn Mortality24 -m LGBMClassifier -l "$LOG" -s 1111 --cpu -gc \
#        -hp execute_repeated_cv.cv_repetitions_to_train=1 execute_repeated_cv.cv_folds_to_train=1
#   3. you land in pdb inside the real pipeline. `git diff` to see/undo your probes.
#
# Places worth a breakpoint (the "how does an individual flow through" path):
#   icu_benchmarks/run.py                      CLI -> gin config -> execute_repeated_cv
#   icu_benchmarks/cross_validation.py         fold split; which stay_ids land in train/val/test
#   icu_benchmarks/data/loader.py              stay -> tensor windows (the per-patient reshape)
#   icu_benchmarks/data/preprocessor.py        recipys steps: impute / scale / outcome scaling
#   icu_benchmarks/data/split_process_data.py  parquet -> splits, the "caching dataset in ram" step
#
# Inspect a cohort directly (shapes/counts only for credentialed sources):
#   bash debug_yaib.sh -c 'import polars as pl,os
#   d=os.environ["COH"]+"/mortality24/miiv"
#   for f in ("sta","dyn","outc"):
#       s=pl.scan_parquet(f"{d}/{f}.parquet"); print(f, s.collect_schema().names()[:8], s.select(pl.len()).collect().item())'
#
# COMPLIANCE: no -wd (W&B). Keep TMPDIR and every output path under $ROOT. mimic_demo /
# eicu_demo cohorts are public; miiv/eicu/hirid are credentialed — shapes yes, rows no.
# =============================================================================
set -uo pipefail

ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
COH="${COH:-$ROOT/ehrapy-usecase/processed_data/yaib-cohorts}"
LOG="${LOG:-$ROOT/ehrapy-usecase/processed_data/yaib-benchmark}"
IMG=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images/nvidia+pytorch+26.03-py3_yaib.sqsh

mkdir -p "$LOG/_logs" "$LOG/_tmp/mpl"

# enroot's ENROOT_DATA_PATH is per-NODE: a fixed container name collides with any other job on
# the same node. Reuse a per-job name if one exists, else create it.
CNAME="yaib_dbg_${SLURM_JOB_ID:-$$}"
if ! enroot list 2>/dev/null | grep -qx "$CNAME"; then
  echo "creating container $CNAME (~40 s, node-local /localscratch) ..."
  enroot create --name "$CNAME" "$IMG" >/dev/null
fi
echo "container: $CNAME   (remove with: enroot remove -f $CNAME)"

ENVS=(--env "MPLCONFIGDIR=$LOG/_tmp/mpl"
      --env "ROOT=$ROOT" --env "YAIB=$YAIB" --env "COH=$COH" --env "LOG=$LOG")
if [ "${GPU:-0}" != "1" ]; then
  # CPU node: skip the enroot nvidia hook. TMPDIR on the shared tree (compliance).
  ENVS+=(--env NVIDIA_VISIBLE_DEVICES=void --env "TMPDIR=$LOG/_tmp")
else
  # GPU: leave the nvidia hook enabled, and TMPDIR MUST be short — deep-model DataLoader
  # workers pass GPU tensors over an AF_UNIX socket under TMPDIR, whose path has a 108-char
  # limit; the shared-tree path overflows it ("AF_UNIX path too long") and the job hangs.
  # Only the ephemeral IPC socket lives there; real outputs still go to -l under $ROOT.
  ENVS+=(--env TMPDIR=/tmp)
  echo "GPU mode: TMPDIR=/tmp (AF_UNIX 108-char limit), nvidia hook enabled"
fi

case "${1:-shell}" in
  shell)   INNER='exec bash' ;;
  python)  INNER='exec python' ;;
  ipython) INNER='command -v ipython >/dev/null && exec ipython || exec python' ;;
  -c)      shift; INNER="exec python -c $(printf '%q' "$*")" ;;
  --)      shift; INNER="exec $*" ;;
  *)       INNER="exec $*" ;;
esac

exec enroot start --mount "$ROOT":"$ROOT" "${ENVS[@]}" "$CNAME" bash -c "
  source /opt/yaib-venv/bin/activate
  cd '$YAIB'
  export PS1='[yaib] \w \$ '
  $INNER
"
