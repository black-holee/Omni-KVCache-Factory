# method=fullkv fastkv snapkv h2o streamingllm gemfilter pyramidinfer
eviction_mode=proportional
tsp_idx=15
tsp_rate=0.2
retain_rate=0.2
# attn_implementation=flash_attention_2
attn_implementation=eager
model_path="/home/ubuntu/data_disk/models/Meta-Llama-3.1-8B-Instruct"
save_dir="outputs/results_squad"

for method in snapkv; do
    CUDA_VISIBLE_DEVICES=0 python -m eval.run_squad \
        --method ${method} \
        --model_path ${model_path} \
        --attn_implementation ${attn_implementation} \
        --save_dir ${save_dir} \
        --eviction_mode ${eviction_mode} \
        --tsp_rate ${tsp_rate} \
        --tsp_idx ${tsp_idx} \
        --retain_rate ${retain_rate}

    # CUDA_VISIBLE_DEVICES=0 python -m eval.eval_squad \
    #     --results_dir ${save_dir}
done