#!/bin/bash

set -euo pipefail


# Setup script for downloading required model checkpoints via gdown.
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --force

FORCE_DOWNLOAD="false"
if [[ "${1:-}" == "--force" ]]; then
  FORCE_DOWNLOAD="true"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[setup] Repo root: ${REPO_ROOT}"

# Check gdown, install if missing.
if ! command -v gdown >/dev/null 2>&1; then
  echo "[setup] gdown not found. Installing with pip..."
  python -m pip install gdown
fi

# Model 1: latent classifier for LDM coupling
LATENT_DST="${REPO_ROOT}/model_weights/ldm/stable-diffusion-2-1-base/base_latent_classifier_resnet_enc_multihead_timecond_False_2025-07-21_01-27-39/checkpoints_latent_classifier/best_latent_classifier.pt"
LATENT_FILE_ID="1YdYA0dIx1N-xRBUiiDXYDKKnc_QleCzU"

# Model 2: image classifier for analysis
IMAGE_DST="${REPO_ROOT}/model_weights/FFHGA_classifier_resnet_enc_multihead_age_group_1024_gender_128/checkpoints_classifier/best_classifier.pt"
IMAGE_FILE_ID="1ar-S6XyYkF9mTu45TSOMVhzyFb3mIFGo"

download_if_needed() {
  local file_id="$1"
  local dst="$2"

  mkdir -p "$(dirname "${dst}")"

  if [[ -f "${dst}" && "${FORCE_DOWNLOAD}" != "true" ]]; then
    echo "[setup] Exists, skip: ${dst}"
    return 0
  fi

  echo "[setup] Downloading -> ${dst}"
  gdown "${file_id}" -O "${dst}"
}

download_if_needed "${LATENT_FILE_ID}" "${LATENT_DST}"
download_if_needed "${IMAGE_FILE_ID}" "${IMAGE_DST}"

echo "[setup] Done."
