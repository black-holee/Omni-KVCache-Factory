#!/bin/bash

# Root Directories

declare -A templete_dict
templete_dict["Qwen3-8B"]="qwen3"
templete_dict["Llama-3.1-8B-Instruct"]="llama3.1"

MODEL_PATH=qwen/Qwen3-8B
# MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

model_name="${MODEL_PATH##*/}"

max_seq_length_list=(4096 8192 16384 32768 65536 131072)
BENCHMARK=synthetic
NUM_SAMPLES=200

# Benchmark and Tasks
source ruler_config_tasks.sh
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

TEMPLATE_TYPE=${templete_dict[$model_name]}

for MAX_SEQ_LENGTH in ${max_seq_length_list[@]}; do
for TASK in ${synthetic[@]}; do

DATA_DIR="./created_data/${TEMPLATE_TYPE}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
mkdir -p ${DATA_DIR}

python -u data/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${MODEL_PATH} \
    --tokenizer_type "hf" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type $TEMPLATE_TYPE \
    --num_samples ${NUM_SAMPLES}
done
done