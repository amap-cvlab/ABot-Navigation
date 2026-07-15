#!/bin/bash
# ============================================================
# 3D Gaussian Splatting — Dynamic Scene Render Server
# ============================================================
# Usage:
#   bash run.sh                              # default args
#   bash run.sh --port 8001                  # custom port
#   bash run.sh --gpus 0,1                   # multi-GPU
#   bash run.sh --scenes_root /path/to/scenes
# ============================================================

set -e

# ---- Environment ----
CONDA_ENV_NAME="abotn_render"
PYTHON="${PYTHON:-python}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Logging ----
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- Verify PyTorch + CUDA ----
if ! ${PYTHON} -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    log_error "PyTorch with CUDA support is required but not detected."
    log_error "Install PyTorch from https://pytorch.org matching your CUDA version."
    exit 1
fi

export PYTHONNOUSERSITE=1
log_info "Python: ${PYTHON} ($(${PYTHON} --version 2>&1 | awk '{print $2}'))"
log_info "PyTorch: $(${PYTHON} -c 'import torch; print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")')"

# ---- Start render server ----
cd "${PROJECT_DIR}"

DEFAULT_ARGS="--port 7036 --gpus 0 --max_scenes_per_gpu 1 --render_scale 1.0 --scenes_root /path/to/3dgs_scenes"

log_info "Starting render server: ${PYTHON} render_sim_dynamic.py ${DEFAULT_ARGS} $@"
${PYTHON} render_sim_dynamic.py ${DEFAULT_ARGS} "$@"
