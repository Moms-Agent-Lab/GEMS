#!/bin/bash

METADATA_FILE="eval/ArtiMuse/evaluation_metadata.jsonl"
EXP_NAME="gems_experiment_v1"
OUTPUT_DIR="eval/ArtiMuse/outputs"

GEN_URL=""
MLLM_URL=""

NUM_WORKERS=32
MAX_ITER=5

echo "Starting GEMS Agent Evaluation..."
echo "Experiment: $EXP_NAME"
echo "Workers: $NUM_WORKERS"

python gen_artimuse.py "$METADATA_FILE" \
    --name "$EXP_NAME" \
    --outdir "$OUTPUT_DIR" \
    --num_workers $NUM_WORKERS \
    --max_iterations $MAX_ITER \
    --gen_url "$GEN_URL" \
    --mllm_url "$MLLM_URL" \
    --agent "gems"

echo "Task Finished!"