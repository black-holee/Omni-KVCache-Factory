method=fullkv
eviction_mode=proportional
tsp_idx=15
tsp_rate=0.6
retain_rate=0.2
window_size=8
attn_implementation=flash_attention_2
model_path="/home/ubuntu/data_disk/models/Meta-Llama-3.1-8B-Instruct"
save_dir="outputs/results_longbench"

for method in fullkv h2o snapkv streamingllm gemfilter fastkv; do
    for retain_rate in 0.2 0.1; do
        CUDA_VISIBLE_DEVICES=0 python -m eval.run_longbench \
            --method ${method} \
            --model_path ${model_path} \
            --attn_implementation ${attn_implementation} \
            --save_dir ${save_dir} \
            --eviction_mode ${eviction_mode} \
            --tsp_rate ${tsp_rate} \
            --tsp_idx ${tsp_idx} \
            --retain_rate ${retain_rate} \
            --window_size ${window_size}
    done
    CUDA_VISIBLE_DEVICES=0 python -m eval.eval_longbench \
        --model_path ${model_path} \
        --results_dir ${save_dir} \
        --retain_rate ${retain_rate} \
        --window_size ${window_size} \
        --tsp_rate ${tsp_rate} \
        --tsp_idx ${tsp_idx}
done