#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${CNNCAP_RAW_DIR:-${PROJECT_ROOT}/data/cnncap/raw}"
OUTPUT_DIR="${PROJECT_ROOT}/data/cnncap"
DATASETS=(55nm_C_2_3_6 55nm_C_2_4_6 15nm_C_2_4_6 15nm_C_2_4_9)

for dataset in "${DATASETS[@]}"; do
    python "${PROJECT_ROOT}/src/prepare_cnncap.py" \
        "${RAW_DIR}/${dataset}.json" \
        --output_dir "${OUTPUT_DIR}/${dataset}"
done
