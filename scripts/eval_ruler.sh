method=fullkv
eviction_mode=proportional
tsp_idx=15
tsp_rate=0.6
retain_rate=0.2
window_size=8
attn_implementation=flash_attention_2

model_path="meta-llama/Llama-3.1-8B-Instruct" # need edit
model_name="${model_path##*/}"
save_dir="outputs/${model_name}/results_ruler"

CUDA_VISIBLE_DEVICES=0

declare -A templete_dict
templete_dict["Qwen3-8B"]="qwen3"
templete_dict["Llama-3.1-8B-Instruct"]="llama3.1"

max_seq_length_list=(4096 8192 16384 32768 65536 131072)

for method in fullkv h2o snapkv streamingllm gemfilter fastkv; do
    for MAX_SEQ_LENGTH in ${max_seq_length_list[@]}; do
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m eval.run_ruler \
            --method ${method} \
            --model_path ${model_path} \
            --attn_implementation ${attn_implementation} \
            --save_dir ${save_dir} \
            --eviction_mode ${eviction_mode} \
            --tsp_rate ${tsp_rate} \
            --tsp_idx ${tsp_idx} \
            --retain_rate ${retain_rate} \
            --window_size ${window_size} \
            --context_length ${MAX_SEQ_LENGTH} \
            --model_template ${templete_dict[$model_name]}

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m eval.eval_ruler \
            --model_path ${model_path} \
            --results_dir ${save_dir} \
            --tsp_rate ${tsp_rate} \
            --tsp_idx ${tsp_idx} \
            --retain_rate ${retain_rate} \
            --window_size ${window_size} \
            --context_length ${MAX_SEQ_LENGTH}
    done
done