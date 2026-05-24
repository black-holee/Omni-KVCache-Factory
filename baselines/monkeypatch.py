import torch
import transformers
import json
import sys

def replace_llama(method):
    llama_module = transformers.models.llama.modeling_llama

    if method == "fastkv":
        from baselines.fastkv.llama_model import (
            LlamaFastKVAttention,
            llama_decoderlayer_forward_fastkv,
            llama_model_forward_fastkv,
        )

        llama_module.LlamaModel.forward = llama_model_forward_fastkv
        llama_module.LlamaDecoderLayer.forward = llama_decoderlayer_forward_fastkv
        if hasattr(llama_module, "LLAMA_ATTENTION_CLASSES"):
            llama_module.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFastKVAttention
        elif hasattr(llama_module, "LlamaFlashAttention2"):
            llama_module.LlamaFlashAttention2 = LlamaFastKVAttention
        else:
            llama_module.LlamaAttention = LlamaFastKVAttention

    elif method == "streamingllm":
        from baselines.streamingllm.llama_model import (
            llama_attn_forward_StreamingLLM,
            llama_flash_attn2_forward_StreamingLLM,
            llama_sdpa_attn_forward_StreamingLLM,
        )

        llama_module.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        if hasattr(llama_module, "LlamaFlashAttention2"):
            llama_module.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        if hasattr(llama_module, "LlamaSdpaAttention"):
            llama_module.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM

    elif method == "h2o":
        from baselines.h2o.llama_model import (
            llama_attn_forward_H2O,
            llama_flash_attn2_forward_H2O,
            llama_sdpa_attn_forward_H2O,
        )

        llama_module.LlamaAttention.forward = llama_attn_forward_H2O
        if hasattr(llama_module, "LlamaFlashAttention2"):
            llama_module.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        if hasattr(llama_module, "LlamaSdpaAttention"):
            llama_module.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O

    elif method == "snapkv":
        from baselines.snapkv.llama_model import (
            llama_attn_forward_SnapKV,
            llama_flash_attn2_forward_SnapKV,
            llama_sdpa_attn_forward_SnapKV,
        )

        llama_module.LlamaAttention.forward = llama_attn_forward_SnapKV
        if hasattr(llama_module, "LlamaFlashAttention2"):
            llama_module.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        if hasattr(llama_module, "LlamaSdpaAttention"):
            llama_module.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV

    elif method == "gemfilter":
        from baselines.gemfilter.llama_model import LlamaGemFilterAttention

        if hasattr(llama_module, "LLAMA_ATTENTION_CLASSES"):
            llama_module.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaGemFilterAttention
        else:
            llama_module.LlamaAttention = LlamaGemFilterAttention

    elif method == "fullkv":
        pass

    else:
        raise NotImplementedError(f"No method found for {method}")

def replace_qwen3(method):
    if method == "fastkv":
        from baselines.fastkv.qwen3_model import qwen3_model_forward_fastkv, qwen3_decoderlayer_forward_fastkv, Qwen3FastKVAttention
        transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = qwen3_model_forward_fastkv
        transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer.forward = qwen3_decoderlayer_forward_fastkv
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention = Qwen3FastKVAttention

    elif method == "streamingllm":
        from baselines.streamingllm.qwen3_model import qwen3_attn_forward_StreamingLLM
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_attn_forward_StreamingLLM
    
    elif method == "h2o":
        from baselines.h2o.qwen3_model import qwen3_attn_forward_H2O
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_attn_forward_H2O

    elif method == "snapkv":
        from baselines.snapkv.qwen3_model import qwen3_attn_forward_SnapKV
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_attn_forward_SnapKV

    elif method == "gemfilter":
        from baselines.gemfilter.qwen3_model import Qwen3GemFilterAttention
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention = Qwen3GemFilterAttention

    # TODO support pyramidinfer in qwen3
    # elif method == "pyramidinfer":
    #     from baselines.pyramidinfer import llama_model
    #     sys.modules["transformers.models.llama.modeling_llama"] = llama_model

    elif method == "fullkv":
        pass

    else:
        raise NotImplementedError(f"No method found for {method}")

def set_model(model, args):
    if args.max_capacity_prompts != -1:
        max_capacity_prompts = args.max_capacity_prompts

    if args.method != "fullkv":
        if args.method in ["streamingllm"]:
            window_size = max_capacity_prompts - 4
        elif args.method in ["gemfilter", "pyramidinfer"]:
            window_size = 1 # does not mean anything actually
        else:
            window_size = args.window_size
            
        kernel_size = args.kernel_size
        pooling = args.pooling
        retain_rate = args.retain_rate

        layers = len(model.model.layers)
        # check if window_size is a list
        if not isinstance(window_size, list):
            window_size = [window_size] * layers
        if not isinstance(max_capacity_prompts, list):
            max_capacity_prompts = [max_capacity_prompts] * layers
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * layers
        if not isinstance(retain_rate, list):
            retain_rate = [retain_rate] * layers

        for i in range(layers):
            model.model.layers[i].self_attn.config.window_size = window_size[i]
            model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
            model.model.layers[i].self_attn.config.kernel_size = kernel_size[i]
            model.model.layers[i].self_attn.config.pooling = pooling
            model.model.layers[i].self_attn.config.merge = args.merge
            model.model.layers[i].self_attn.config.retain_rate = retain_rate[i]
            model.model.layers[i].self_attn.config.eviction_mode = args.eviction_mode

        # FastKV
        if args.method == "fastkv":
            from baselines.fastkv.utils import compress_fastkv
            args.window_size = window_size
            args.kernel_size = kernel_size
            compress_fastkv(model, args)

        elif args.method == "gemfilter":
            from baselines.gemfilter.utils import set_topk
            set_topk(model, args, mode='gemfilter')

        elif args.method  == "pyramidinfer":
            if "llama" in args.model_path.lower():
                if args.retain_rate == 0.35:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/llama31_8b_35%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.01
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.01
                elif args.retain_rate == 0.5:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/llama31_8b_50%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.3
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2
                elif args.retain_rate == 0.6:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/llama31_8b_60%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.7
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2
                else:
                    raise NotImplementedError(f"No config found for retain_rate={args.retain_rate}")
            elif "ministral" in args.model_path.lower():
                if args.retain_rate == 0.35:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/ministral_8b_35%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.01
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.01
                elif args.retain_rate == 0.6:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/ministral_8b_60%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.75
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2
                else:
                    raise NotImplementedError(f"No config found for retain_rate={args.retain_rate}")
            elif "nemo" in args.model_path.lower():
                if args.retain_rate == 0.6:
                    args.pyramidinfer_config = "baselines/pyramidinfer/pyramidinfer_configs/nemo_12b_60%.json"
                    pyramidinfer_config = json.load(open(args.pyramidinfer_config))
                    assert pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] == 0.78
                    assert pyramidinfer_config["prefill_stage"]["recent_ratio"] == 0.2

            from baselines.pyramidinfer.utils import load_pyramid_config
            model = load_pyramid_config(model, pyramidinfer_config)