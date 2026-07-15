#!/usr/bin/env bash
# Wrapper to start the render server.
# Delegates to render_server/run.sh.
#
# Edit the variables below before running.
#
# ── Usage ──
#   bash scripts/start_POIGoal_render_server.sh

set -e
cd "$(dirname "$0")/.."

# ============================================================
# Render server parameters
# ============================================================

# Port the render server listens on
PORT="7036"

# GPU(s) to use, e.g. "0" or "0,1,2"
GPUS="0"

# Maximum number of scenes loaded on each GPU
MAX_SCENES_PER_GPU="1"

# Render scale for super-sampling anti-aliasing
RENDER_SCALE="1.5"

# Root directory containing the 3DGS scene folders
SCENES_ROOT="/path/to/poigoal_scenes"

# ============================================================
# Launch render server
# ============================================================

RENDER_SERVER_SCRIPT="render_server/run.sh"

if [ ! -f "${RENDER_SERVER_SCRIPT}" ]; then
    echo "[ERROR] Render server script not found: ${RENDER_SERVER_SCRIPT}"
    exit 1
fi

bash "${RENDER_SERVER_SCRIPT}" \
    --port "${PORT}" \
    --gpus "${GPUS}" \
    --max_scenes_per_gpu "${MAX_SCENES_PER_GPU}" \
    --render_scale "${RENDER_SCALE}" \
    --scenes_root "${SCENES_ROOT}"
