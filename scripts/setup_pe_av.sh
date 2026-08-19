#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PE_AV_DIR="${PE_AV_DIR:-${ROOT}/third_party/perception_models}"

mkdir -p "$(dirname "${PE_AV_DIR}")"

if [[ ! -d "${PE_AV_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/perception_models.git "${PE_AV_DIR}"
else
  git -C "${PE_AV_DIR}" fetch --all --tags
fi

python -m pip install -e "${PE_AV_DIR}"

echo "PE-AV is available at ${PE_AV_DIR}"
