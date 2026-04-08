#!/bin/bash
timestamp=$(date +%Y%m%d%H%M%S)
target="asap7"

config=configs/$target.json

cmd="bin/pattern_gen \
-i itf_lib/$target.ctf \
-o pattern_gen_output/${target}_${timestamp}.jsonl \
-n 96 \
-p 0.005 \
--merge-diel false \
--param-json $config \
--seed 1"

echo "Running command: $cmd"
$cmd 2>&1 | tee pattern_gen_output/${target}_${timestamp}.log