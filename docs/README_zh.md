# Omni-KVCache-Factory

<p align="center">
  <img src="images/logo.png" alt="Omni-KVCache-Factory Logo">
</p>

Omni-KVCache-Factory: 一个汇总 KVCache 优化方法的代码仓库，提供各种方法可直接使用的代码，支持多种模型和基准测试，帮助研究人员快速运行实验和验证想法。

## 使用方法
### 1. 安装
使用 requirements 包进行安装。
```
conda create -n okfc python=3.10.5
conda activate okfc
cd Omni-KVCache-Factory
pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ./flash_attn-2.7.2.post1+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 2. 快速开始
使用 KVCache 方法进行推理，并支持 LongBench、Ruler、Needle-in-a-Haystack 以及速度基准测试的评估。

```
# 运行 LongBench 评估
./scripts/eval_longbench.sh

# 运行 Needle-in-a-Haystack 评估
./scripts/eval_needle.sh

# 运行 RULER 评估
# 1. 准备 RULER 数据
cd benchmarks/RULER
./ruler_prepare_data.sh

# 2. 运行评估
cd ../..
./scripts/eval_ruler.sh
```

## 支持的功能

### 纯文本
- **模型类型**
  - qwen3 (`Qwen3-0.6B`, `Qwen3-8B`, `Qwen3-14B` ...)
- **KV 优化方法**
  - H2O ([H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html))
  - StreamingLLM ([Efficient Streaming Language Models with Attention Sinks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html))
  - SnapKV ([SnapKV: LLM Knows What You are Looking for Before Generation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html))
  - GemFilter ([Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction](https://arxiv.org/abs/2409.17422))
  - FastKV ([FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](https://arxiv.org/abs/2502.01068))
- **基准测试**
  - [LongBench](https://github.com/THUDM/LongBench)
  - [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
  - [RULER](https://github.com/NVIDIA/RULER/tree/main)

### 多模态
- [ ] 暂不支持，后续计划支持

## TODO
### 模型类型
- [ ] llama
- [ ] mistral

### KV 优化方法
- [ ] PyramidInfer
- [ ] ASL

### 基准测试
- [ ] InfiniteBench
