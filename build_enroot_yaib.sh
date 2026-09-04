#!/usr/bin/env bash
# =============================================================================
# Build a stable YAIB enroot container and export it as a persistent image.
#
# Pattern (per ~/script_templates): enroot containers live on node-local
# /localscratch (fast create ~25s, but EPHEMERAL). Persistent code/data come from
# bind-mounts; a persistent ENVIRONMENT is baked into a `_yaib.sqsh` image on the
# ml01 workspace (same convention as the existing _pdac / _ehrdata_ehrapy images).
#
# The image ships Python 3.12 / numpy 2.1 / torch 2.11a0 which clash with YAIB
# (py3.10 / torch 2.6.0+cu118 / numpy 1.24.3). So we build a clean uv-managed
# **python 3.10 venv** at /opt/yaib-venv inside the container and install YAIB there.
#
# Run on a COMPUTE node (this session's node is fine). ~15-25 min (torch download + export).
# =============================================================================
set -euo pipefail

ROOT=/ictstr01/groups/shared/physionet-credentialized
YAIB=$ROOT/ehrapy-usecase/code/yaib
IMAGES=/ictstr01/groups/ml01/workspace/eljas.roellin/enroot_images
BASE_IMG=$IMAGES/nvidia+pytorch+26.03-py3.sqsh
OUT_IMG=$IMAGES/nvidia+pytorch+26.03-py3_yaib.sqsh
CONTAINER=yaib
VENV=/opt/yaib-venv

export NVIDIA_VISIBLE_DEVICES=void   # CPU node: skip nvidia GPU hook
export ENROOT_MOUNT_HOME=n           # isolate from $HOME so everything bakes into the rootfs

# 1) fresh container on the fast localscratch default path
if enroot list 2>/dev/null | grep -qx "$CONTAINER"; then enroot remove -f "$CONTAINER"; fi
echo "[build] creating container '$CONTAINER' from $(basename "$BASE_IMG") ..."
enroot create --name "$CONTAINER" "$BASE_IMG"

# 2) build the py3.10 venv INSIDE the rootfs (as root, so /opt + /usr/local are writable), install YAIB
echo "[build] installing YAIB into $VENV (python 3.10) ..."
enroot start --root --rw \
  --mount "$ROOT":"$ROOT" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env UV_PYTHON_INSTALL_DIR=/opt/uv/python \
  --env UV_CACHE_DIR=/opt/uv/cache \
  --env YAIB="$YAIB" --env VENV="$VENV" \
  "$CONTAINER" bash -euxc '
    pip install --quiet --root-user-action=ignore uv
    uv venv --python 3.10 "$VENV"
    UV_IDX="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu118"
    uv pip install --python "$VENV/bin/python" $UV_IDX -r "$YAIB/requirements.txt"
    uv pip install --python "$VENV/bin/python" $UV_IDX -e "$YAIB"
    chmod -R a+rX /opt/uv "$VENV"
    "$VENV/bin/python" -c "import icu_benchmarks, lightgbm, torch, numpy; print(\"import OK — numpy\", numpy.__version__, \"torch\", torch.__version__)"
    "$VENV/bin/icu-benchmarks" -h >/dev/null && echo "icu-benchmarks CLI OK"
  '

# 3) export the customized container to a persistent image (workspace)
echo "[build] exporting -> $OUT_IMG ..."
[ -f "$OUT_IMG" ] && mv -f "$OUT_IMG" "$OUT_IMG.prev"
enroot export --output "$OUT_IMG" "$CONTAINER"
ls -lh "$OUT_IMG"
echo "[build] DONE. Persistent YAIB image: $OUT_IMG"
echo "        venv inside: $VENV  (activate: source $VENV/bin/activate, or call $VENV/bin/icu-benchmarks)"
