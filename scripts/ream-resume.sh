#!/bin/bash
# REAM Resume Script
# 
# This script demonstrates how to run REAM with checkpointing support.
# It can be safely interrupted (SIGTERM/SIGINT) and will save a checkpoint.
# Resume by running the same command again with --auto_resume or --resume_from_checkpoint.
#
# Usage:
#   ./ream-resume.sh                    # Start new run or auto-resume
#   ./ream-resume.sh --resume_from_checkpoint /path/to/checkpoint  # Resume from specific checkpoint
#   NUM_GPUS=4 ./ream-resume.sh         # Use 4 GPUs in parallel
#
# For interruptible cloud instances (2/3 the price of continuous):
#   1. Start this script on an interruptible instance
#   2. If interrupted, it will save a checkpoint and exit
#   3. Restart the script to resume from the checkpoint
#   4. Repeat until complete
#
# For multi-GPU speedup:
#   - Set NUM_GPUS environment variable to use multiple GPUs in parallel
#   - Each GPU gets a copy of the model and processes calibration data in parallel
#   - Provides near-linear speedup (e.g., 4 GPUs = ~4x faster)

set -e

# Default arguments
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B}"
DATASET="${DATASET:-ream_mixed}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-1}"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume_from_checkpoint)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --auto_resume)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Run REAM with checkpointing enabled
python -m ream.ream \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --compression_ratio "$COMPRESSION_RATIO" \
    --seed "$SEED" \
    --num_gpus "$NUM_GPUS" \
    --auto_resume \
    "${EXTRA_ARGS[@]}"

echo "REAM completed successfully!"
