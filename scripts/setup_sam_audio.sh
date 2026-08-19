#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_AUDIO_DIR="${SAM_AUDIO_DIR:-${ROOT}/third_party/sam-audio}"

mkdir -p "$(dirname "${SAM_AUDIO_DIR}")"

if [[ ! -d "${SAM_AUDIO_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/sam-audio "${SAM_AUDIO_DIR}"
else
  git -C "${SAM_AUDIO_DIR}" fetch --all --tags
fi

python -m pip install -e "${SAM_AUDIO_DIR}"

echo "SAM-Audio is available at ${SAM_AUDIO_DIR}"

