# YAIB baseline — self-serve RUNBOOK (enroot container)

Phase 1 step (b) of the OMOP/ehrdata/YAIB suite: run the YAIB benchmark on the ricu
cohorts extracted by repo (a) `yaib-cohorts`. This is the **run-it-yourself** guide.

**Status: environment BUILT + pipeline VALIDATED (2026-07-11), re-verified 2026-08-31.** You do
**not** need to rebuild anything — the stable environment is a persistent enroot image. Recreate
the container from it (~30 s) and run.

2026-08-31 re-check: image intact; py3.10.20 / numpy 1.24.3 / torch 2.6.0+cu118 / lgbm 4.6.0;
`icu_benchmarks` resolves to this checkout (editable). Two runs via `sbatch run_yaib.sbatch`, both
1 rep x 1 fold, seed 1111, CPU:

| run | result | cross-check |
|---|---|---|
| miiv / mortality24 / `LGBMClassifier` (defaults) | **AUROC 0.8743 / AUPRC 0.4006** | matches the §9 record 0.874 / 0.401 exactly |
| miiv / los / `LGBMRegressor` (`TASK=los`) | MAE 0.2298 normalized -> **38.6 h** (x168); R2 0.340 | inside the paper's miiv range 39.0-40.6, alongside the §10 DL band 36.3-38.1 |

The LoS run doubles as the check that `-t RegressionLoS` is wired up: under plain `Regression.gin`
(`outcome_max=15`) the denormalized MAE would be nonsense. `los` takes ~10 min for one fold vs
~4.5 min for `mortality24` — the 168 h horizon makes a much larger `dyn` table.

---

## 0. Paths (copy this block; every command below uses these)

```bash
export NVIDIA_VISIBLE_DEVICES=void          # CPU node: skip the nvidia GPU hook
ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
COH=$ROOT/ehrapy-usecase/processed_data/yaib-cohorts        # inputs  (repo a output)
LOG=$ROOT/ehrapy-usecase/processed_data/yaib-benchmark      # outputs (this repo)
IMG=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images/nvidia+pytorch+26.03-py3_yaib.sqsh
mkdir -p "$LOG/_logs" "$LOG/_tmp"
```

- Container venv (python 3.10, numpy 1.24.3, torch 2.6.0+cu118): `/opt/yaib-venv` inside the image.
- `enroot` 3.5.0. The cluster is **CPU-only** (`cpu_p` / `interactive_cpu_p`) → always pass `--cpu`.

---

## 1. Get a CPU allocation

The enroot container lives on the node's **ephemeral** `/localscratch`, so it dies with the
job. That's fine — recreating it from the image (step 2) takes ~40 s. **Use a batch job for
long runs** so a timeout can't kill you mid-run (that's what happened before).

Interactive (quick tests):
```bash
srun -p interactive_cpu_p -c 8 --mem 32G -t 04:00:00 --pty bash
```
Batch (real runs) — see the sbatch template in §6.

---

## 2. Recreate the container from the persistent image (~40 s)

```bash
if enroot list 2>/dev/null | grep -qx yaib; then enroot remove -f yaib; fi
enroot create --name yaib "$IMG"
enroot list          # should show: yaib
```

## 3. Self-test (10 s) — confirms imports + CLI

```bash
enroot start --mount "$ROOT":"$ROOT" --env NVIDIA_VISIBLE_DEVICES=void yaib bash -c \
  'source /opt/yaib-venv/bin/activate && \
   python -c "import icu_benchmarks, lightgbm, torch, numpy; print(\"OK\", numpy.__version__, torch.__version__)" && \
   icu-benchmarks -h >/dev/null && echo "CLI OK"'
```
Expect: `OK 1.24.3 2.6.0+cu118` and `CLI OK`.

---

## 4. Run a benchmark

**Generic template** (fill in `<task-dir>`, `<TaskType>`, `<Label>`, `<Model>`):

```bash
enroot start --mount "$ROOT":"$ROOT" \
  --env NVIDIA_VISIBLE_DEVICES=void --env TMPDIR="$LOG/_tmp" \
  yaib bash -c "source /opt/yaib-venv/bin/activate && cd '$YAIB' && \
    icu-benchmarks train -d '$COH/<task-dir>/<src>' -n <src> \
      -t <TaskType> -tn <Label> -m <Model> -l '$LOG' -s 1111 --cpu -gc" \
  > "$LOG/_logs/<task-dir>_<src>_<Model>_$(date +%Y%m%dT%H%M%S).log" 2>&1
```

Flag notes: `-d` cohort dir · `-n` source name · `-t` task **gin** (must be exact) · `-tn`
free-form experiment label · `-m` model gin · `-l` log dir · `-s` seed · `--cpu` (required
here) · `-gc` generate preprocessing cache. **Never add `-wd`** (that turns on W&B → off-site
egress of cohort-derived data; forbidden — see §7).

### Cohort matrix (inputs under `$COH/<task>/<src>/`)

| task dir            | `-t` TaskType         | suggested `-tn` | sources                 |
|---------------------|-----------------------|-----------------|-------------------------|
| `mortality24`       | `BinaryClassification`| `Mortality24`   | miiv · eicu · hirid     |
| `aki`               | `BinaryClassification`| `Aki`           | miiv · eicu · hirid     |
| `sepsis`            | `BinaryClassification`| `Sepsis`        | miiv · eicu · hirid     |
| `kidney_function`   | `Regression`          | `KidneyFunction`| miiv · eicu · hirid     |
| `los`               | `RegressionLoS`       | `LengthOfStay`  | miiv · eicu · hirid     |

(`mimic_demo` also present for public-data smoke tests; `base` = plain cohort, no task label.)

⚠️ `los` needs **`RegressionLoS`**, not `Regression`: the latter hardcodes `outcome_max=15`
(kidney function, mg/dL) and mis-scales a [0,168] h outcome. See §10.

Models (`-m`, from `configs/prediction_models/`): `LGBMClassifier`, `LGBMRegressor`,
`LogisticRegression`, `ElasticNet`, `GRU`, `LSTM`, `TCN`, `Transformer`, `RNN`, ...
Start with **LGBM / LogReg** (fast on CPU); deep models are slow here.

### Fast smoke test (1 fold, ~6 min) vs full run

`-hp` passes **arbitrary gin bindings**. Default is **5 reps × 5 folds = 25 trainings**.
For a quick single-fold sanity check add:
```bash
      ... -m LGBMClassifier -l '$LOG' -s 1111 --cpu -gc \
      -hp execute_repeated_cv.cv_repetitions_to_train=1 execute_repeated_cv.cv_folds_to_train=1
```
- 1 rep × 1 fold ≈ **6 min** (one held-out AUROC).
- 1 rep × 5 folds ≈ **30 min** (mean±std over folds).
- full 5 × 5 ≈ **~2.5 h** (paper-comparable mean±std).
- **Bottleneck is per-fold "caching dataset in ram" (~5 min); the LGBM fit is ~8 s.** So cost
  scales with number of folds, not model complexity, for tree models. Hyperparameter search
  (Optuna) runs **only** if you add `--tune` (off by default).

### Deep models on GPU (Transformer, GRU, LSTM, TCN)

There **are** GPUs (`gpu_p` qos `gpu_normal`; also `interactive_gpu_p`). Submit a GPU job —
ready-made template: **`$YAIB/smoke_gpu_transformer.sbatch`** (`sbatch smoke_gpu_transformer.sbatch`).
Key differences from the CPU path:
- **Omit `--cpu`** (Lightning auto-selects the GPU; verify with the `torch.cuda.is_available()` line the script prints).
- **Do NOT set `NVIDIA_VISIBLE_DEVICES=void`** — that disables the GPU. The enroot hook
  `98-nvidia.sh` + the image's baked `NVIDIA_VISIBLE_DEVICES=all` expose it automatically.
- **Set `--env TMPDIR=/tmp`** (short, node-local). ⚠️ Deep-model DataLoader workers share GPU
  tensors over an **AF_UNIX socket under `$TMPDIR`**; the long shared-tree path overflows the
  108-char socket limit → `OSError: AF_UNIX path too long` and the job hangs until it times out.
  A short `TMPDIR` fixes it. (Regular `-l` outputs are unaffected — keep those in the shared tree.)
- DL defaults (`DLCommon.gin`): 50 epochs, batch 64, early-stop patience 10. Verified:
  Transformer / miiv / mortality24, 1 fold, no tune → AUROC 0.861 in ~7.5 min on a V100.
- Harmless: on older V100 nodes the NGC image prints `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` /
  "GPU not supported" banners; torch still runs on the GPU (`GPU used: True`). Pin newer GPUs
  via `--gres`/`--constraint` if you prefer.

---

## 5. Read the results

Outputs go to `$LOG/<src>/<Label>/<Model>/<timestamp>/`:
- `repetition_*/fold_*/` — per-fold model (`last.joblib`), metrics, `durations.json`.
- Aggregated metrics are printed to the run log (`$LOG/_logs/*.log`) as
  `Accumulated results: {'avg': {'AUC': ..., 'PR': ...}, 'std': {...}}`.

Grep a finished run's headline number:
```bash
grep -E "Accumulated results" "$LOG/_logs/<your-run>.log"
```
`AUC` = AUROC, `PR` = AUPRC (binary tasks). Regression tasks report MAE/R² instead.
Benign log noise you can ignore: `Failed to save/aggregate shap values` (SHAP export is
optional), and `Trial 0 ... value: 0.0` when `--tune` is off.

---

## 6. `run_yaib.sbatch` — the batch entry point (CPU)

**It exists now** (it used to be a copy-paste template). Everything is env-driven, so one script
covers the whole matrix and you never edit the file:

```bash
cd "$YAIB"
sbatch run_yaib.sbatch                                          # miiv/mortality24/LGBM, 1x1 smoke
TASK=aki SRC=hirid                    sbatch --export=ALL run_yaib.sbatch
TASK=los SRC=miiv REPS=5 FOLDS=5      sbatch --export=ALL run_yaib.sbatch   # paper-comparable
COH=<other cohort dir>                sbatch --export=ALL run_yaib.sbatch   # e.g. the OMOP route

# whole classification matrix, one job each:
for S in miiv eicu hirid; do for T in mortality24 aki sepsis; do
  TASK=$T SRC=$S sbatch --export=ALL -J yaib_${T}_${S} run_yaib.sbatch; done; done
```

| var | default | notes |
|---|---|---|
| `TASK` | `mortality24` | one of the five; sets `-t`/`-tn` from the table in §4 |
| `SRC` | `miiv` | `miiv` · `eicu` · `hirid` (· `mimic_demo`) |
| `MODEL` | `LGBMClassifier` / `LGBMRegressor` | per task |
| `REPS` / `FOLDS` | `1` / `1` | 1x1 ≈ 6 min · 1x5 ≈ 30 min · 5x5 ≈ 2.5 h |
| `SEED` | `1111` | |
| `COH` / `LOG` | the standard dirs | point `COH` elsewhere to benchmark other cohorts |

Baked in: `los` → `-t RegressionLoS` (plain `Regression.gin` hardcodes `outcome_max=15`, right
only for kidney function); `CNAME=yaib_$SLURM_JOB_ID` so parallel jobs on one node don't clobber
each other's rootfs; never `-wd`. Defaults `cpu_p`/`cpu_normal`, 8 cores, 64 G, 12 h.

Deep models on GPU → `run_dl_matrix.sbatch` (§10), not this script.

---

## 6b. Debugging / understanding the code — `debug_yaib.sh`

```bash
bash debug_yaib.sh                  # bash shell in the container: venv active, cwd=$YAIB
bash debug_yaib.sh ipython          # or python
bash debug_yaib.sh -c '<snippet>'   # one-off
bash debug_yaib.sh -- <any cmd>
GPU=1 bash debug_yaib.sh            # inside a gpu_p job
```

The container installs `icu_benchmarks` **editable against this checkout** — verified:
`import icu_benchmarks` resolves to `$YAIB/icu_benchmarks`. So repo edits take effect on the next
run with no rebuild, and the debugging loop is:

1. drop `breakpoint()` into the file you care about,
2. `bash debug_yaib.sh`, then run the `icu-benchmarks train ...` line at the prompt,
3. you land in `pdb` inside the real pipeline. `git diff` shows your probes; `git checkout` removes them.

Where to put the breakpoint to follow an individual through the analysis:

| file | what happens there |
|---|---|
| `icu_benchmarks/run.py` | CLI → gin config → `execute_repeated_cv` |
| `icu_benchmarks/cross_validation.py` | fold split — which `stay_id`s land in train/val/test |
| `icu_benchmarks/data/split_process_data.py` | parquet → splits; the ~5 min "caching dataset in ram" |
| `icu_benchmarks/data/preprocessor.py` | recipys steps: impute / scale / **outcome** scaling (the §10 bug lived here) |
| `icu_benchmarks/data/loader.py` | stay → tensor windows, i.e. the per-patient reshape |

Cohort shapes without touching rows (`sta`/`dyn`/`outc` per task dir) — measured for
`mortality24/miiv`: `sta` 49,523 x 5 · `dyn` 1,238,075 x 50 (48 vars + `stay_id` + `time`) ·
`outc` 49,523 x 2.

The script uses a per-job container name (`yaib_dbg_$SLURM_JOB_ID`) and reuses it across calls;
drop it with `enroot remove -f yaib_dbg_<id>`.

---

## 7. Compliance (non-negotiable)

- **Credentialed patient data** (MIMIC-IV / eICU / HiRID). Never open the raw data files;
  never let data / derived data / logs leave the shared tree — **no `/tmp`, no scratchpad**.
  `TMPDIR` and every `-l`/output path must be under `$ROOT`.
- **No online W&B** (`-wd`) — it would push cohort-derived data off-site.

## 8. If the persistent image is ever lost — rebuild (~15–25 min)

Only if `$IMG` disappears. Re-run the build script (idempotent):
```bash
bash "$YAIB/build_enroot_yaib.sh"     # base image + py3.10 uv venv + pip install -e . + enroot export
```

---

## 9. Validation report (2026-07-11)

Recovered after the prior interactive job hit a SLURM timeout mid-run. The persistent image
`nvidia+pytorch+26.03-py3_yaib.sqsh` (26 G, built 2026-07-10 19:21) survived intact; the live
container + the in-flight run were recreated/re-run.

**Smoke test — `miiv` / `mortality24` / `LGBMClassifier`, 1 rep × 1 fold, seed 1111, CPU:**

| metric        | value |
|---------------|-------|
| AUROC (`AUC`) | 0.874 |
| AUPRC (`PR`)  | 0.401 |
| val logloss   | 0.185 (early-stopped @ iter 178) |
| wall time     | 6 m 36 s (≈5 min = dataset RAM-caching) |

AUROC 0.874 matches YAIB's published LGBM mortality baseline (~0.87) → the full chain
(recreated container → ricu cohort format → training/eval) reproduces correctly. Full 5×5 CV
for the paper-comparable mean±std was **not** run (stopped here by choice).

Housekeeping: a partial killed run remains at
`$LOG/miiv/Mortality24/LGBMClassifier/2026-07-10T20-04-26/` (incomplete) — safe to delete.

---

## 10. DL model × task baseline matrix on miiv (2026-07-12)

All 4 paper DL models × all 5 tasks on **MIMIC-IV (miiv)**, **1 rep × 1 fold**, GPU, paper's
chosen HPs (Tables 24–27, MIMIC-IV rows; fixed defaults Table 22). No tuning, no W&B. Seed 1111.
Script: **`$YAIB/run_dl_matrix.sbatch`**. Per-run logs: `$LOG/_logs/dlmatrix_<MODEL>_<task>.log`;
summaries: `$LOG/_logs/dlmatrix_<MODEL>_summary.tsv`.

**Classification — AUROC / AUPRC:**

| task         | Transformer   | GRU           | LSTM          | TCN           |
|--------------|---------------|---------------|---------------|---------------|
| mortality24  | 0.862 / 0.368 | 0.857 / 0.386 | 0.828 / 0.309 | 0.864 / 0.378 |
| aki          | 0.903 / 0.695 | 0.899 / 0.707 | 0.877 / 0.653 | 0.888 / 0.678 |
| sepsis       | 0.803 / 0.072 | 0.830 / 0.092 | 0.802 / 0.081 | 0.804 / 0.083 |

**Regression — MAE in physical units** (after the 2026-07-12 double-scaling bugfix; KF via
`-t Regression` scale 15 → mg/dL, LoS via `-t RegressionLoS` scale 168 → hours. Denormalize as
`MAE_physical = MAE_normalized × scale`):

| task (unit)              | Transformer | GRU   | LSTM  | TCN   | paper miiv  |
|--------------------------|-------------|-------|-------|-------|-------------|
| kidney_function (mg/dL)  | 0.291       | 0.280 | 0.269 | 0.277 | 0.28–0.32   |
| los (hours)              | 36.3        | 36.5  | 38.1  | 37.0  | 39.0–40.6   |

These now match the paper. ⚠️ **Before** the fix, `_process_outcome` double-scaled the outcome
(KF `÷225` not `÷15`), so DL models undertrained (tiny loss) and collapsed to ≈the mean — the
normalized MAE looked tiny (~0.003) but denormalized to a nonsensical ~0.05 mg/dL. See the
double-scaling gotcha below. The raw normalized MAEs from the fixed runs are
`kidney_function ≈ 0.018–0.019`, `los ≈ 0.216–0.227`.

Sanity: mortality ~0.83–0.86, AKI ~0.88–0.90, sepsis hardest ~0.80–0.83 — all consistent with
YAIB's published ranges. Single-fold numbers (high variance); use 5×5 CV for paper-comparable ±std.

### Reproduce

```bash
cd "$YAIB"
# one job per model (each loops its 5 tasks, ~30 min except TCN):
for M in Transformer GRU LSTM; do sbatch -J dl_$M --export=ALL,MODEL=$M run_dl_matrix.sbatch; done
# TCN is ~15-40x slower (~75 min/task) -> run one task per job with a longer limit, in parallel:
for T in mortality24 aki sepsis kidney_function los; do
  sbatch -J tcn_$T --time=03:00:00 --export=ALL,MODEL=TCN,ONLYTASK=$T run_dl_matrix.sbatch
done
```
The paper HP values for each model×task (MIMIC-IV) are baked into `run_dl_matrix.sbatch`
(case on `$MODEL`). `ONLYTASK=<task>` runs a single task; unset = all 5.

### Gotchas learned (baked into the script)

- **Unique container name per job.** enroot's `ENROOT_DATA_PATH` is **per-node**, not per-job.
  Two jobs on the same node using the same container name (`yaib`) clobber each other's rootfs
  and both die in ~40 s (`dir_scan: File exists` / `No such file or directory`). The script uses
  `CNAME=yaib_$SLURM_JOB_ID`. **Apply this to any parallel submissions** (the §6 template's fixed
  `yaib` name is fine only for a single job at a time).
- **TCN is slow** (~75 min/task on GPU; doesn't early-stop quickly). Don't loop 5 TCN tasks in one
  2 h job — it times out after ~1. Run per-task with `--time=03:00:00` (see above).
- **`-hp` = gin bindings** applied after the model+task gin (`run.py:142`), so scalar `-hp` values
  override the tuple ranges; when every HP is a scalar the untuned Optuna trial logs
  `parameters: {}` = fully deterministic (no sampling).
- **Regression outcome double-scaling (bug, fixed 2026-07-12).** `PolarsRegressionPreprocessor._process_outcome`
  did `outcome_rec.prep()` then `outcome_rec.bake()`; recipys `prep()` already fits+bakes, so `bake()`
  re-applied the affine map → outcomes scaled by `(max−min)` **twice** (KF `÷225` not `÷15`) →
  DL models undertrain and collapse to the mean. Fix: use `prep()`'s return, drop `bake()`
  (both Polars + Pandas variants). Separately, `Regression.gin` hardcodes `outcome_max=15` (right
  only for KF mg/dL); LoS is [0,168] h so use **`-t RegressionLoS`** (added). Denormalize MAE with
  KF ×15 (mg/dL), LoS ×168 (hours). Regression runs log `loss`, `MAE`, `R2` and `RMSE` — an
  earlier note here said `loss`+`MAE` only; corrected 2026-08-31 against a live LGBMRegressor run.
