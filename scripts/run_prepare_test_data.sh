#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${PROJECT_ROOT}/pattern_gen/pattern_gen_output/asap7_test.jsonl"
OUTPUT="${PROJECT_ROOT}/data/asap7_test"

python "${PROJECT_ROOT}/src/prepare.py" \
	"$INPUT" \
	--output_dir "$OUTPUT" \
	--train_ratio 1

rm "$OUTPUT/test_data.pt"
rm "$OUTPUT/val_data.pt"
mv "$OUTPUT/train_data.pt" "$OUTPUT/test_data.pt"
