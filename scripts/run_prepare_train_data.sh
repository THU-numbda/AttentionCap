#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${PROJECT_ROOT}/pattern_gen/pattern_gen_output/asap7.jsonl"
OUTPUT="${PROJECT_ROOT}/data/asap7"

python "${PROJECT_ROOT}/src/prepare.py" \
	"$INPUT" \
	--output_dir "$OUTPUT"
