#!/bin/bash
#
# REAM: Router-weighted Expert Activation Merging
#
# Usage:
#   bash experiments/ream-cli.sh <CUDA_DEVICES> [MODEL_NAME] [SEED] [COMPRESSION_RATIO] [MAX_CLUSTER_SIZE] [RUN_LM_EVAL] [RUN_EVALPLUS] [RUN_LIVE_CODE_BENCH] [RUN_MATH]
#
# Examples:
#   bash experiments/ream-cli.sh 0
#   bash experiments/ream-cli.sh 0,1 Qwen/Qwen3-30B-A3B
#   bash experiments/ream-cli.sh 0 Qwen/Qwen3-30B-A3B 42 0.25 16

set -euo pipefail

CUDA_DEVICES=${1}
MODEL_NAME=${2:-"Qwen/Qwen3-30B-A3B"}
SEED=${3:-42}
COMPRESSION_RATIO=${4:-0.25}
MAX_CLUSTER_SIZE=${5:-16}
RUN_LM_EVAL=${6:-true}
RUN_EVALPLUS=${7:-true}
RUN_LIVE_CODE_BENCH=${8:-false}
RUN_MATH=${9:-false}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

short_model_name=$(echo $MODEL_NAME | cut -d'/' -f2)
server_log_file_name="ream_${short_model_name}_${SEED}.log"
port=8000

echo "============================================="
echo "REAM: Router-weighted Expert Activation Merging"
echo "============================================="
echo "Model:             ${MODEL_NAME}"
echo "Seed:              ${SEED}"
echo "Compression ratio: ${COMPRESSION_RATIO}"
echo "Max cluster size:  ${MAX_CLUSTER_SIZE}"
echo "CUDA devices:      ${CUDA_DEVICES}"
echo "============================================="

python src/reap/ream.py \
    --compression-ratio ${COMPRESSION_RATIO} \
    --model-name ${MODEL_NAME} \
    --dataset-name ream_mixed \
    --merge-method frequency_weighted_average \
    --permute activation_weight \
    --cluster-method agglomerative \
    --profile false \
    --server_log_file_name $server_log_file_name \
    --vllm-port $port \
    --expert-sim router_logits \
    --distance_measure cosine \
    --frequency-penalty false \
    --merged-model-dir-name "ream-${COMPRESSION_RATIO}" \
    --cluster-description "ream" \
    --max-cluster-size ${MAX_CLUSTER_SIZE} \
    --do-eval true \
    --run-lm-eval ${RUN_LM_EVAL} \
    --run-evalplus ${RUN_EVALPLUS} \
    --run-livecodebench ${RUN_LIVE_CODE_BENCH} \
    --run-math ${RUN_MATH} \
    --seed ${SEED}
