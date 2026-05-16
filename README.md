# Omni-KVCache-Factory

<p align="center">
  <img src="docs/images/logo.png" alt="Omni-KVCache-Factory Logo">
</p>

Omni-KVCache-Factory: A repository summarizing KVCache optimization methods, providing ready-to-use code for various methods, supporting multiple models and benchmarks to help researchers quickly run experiments and validate ideas.

<p align="center"><a href="docs/README_zh.md">📖 [简体中文] <a href="docs/DevelopmentGuide.md">🔧 [开发指南]</a></p>

## Usage
### 1. Installation
Installation with the requirements package.
```
conda create -n okfc python=3.10.5
conda activate okfc
cd Omni-KVCache-Factory
pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ./flash_attn-2.7.2.post1+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 2. Quick Start
Inference with KVCache methods and evaluation for LongBench, Ruler, Needle-in-a-Haystack, and speedup benchmark.

```
# Run LongBench Evaluation
./scripts/eval_longbench.sh

# Run Needle-in-a-Haystack Evaluation
./scripts/eval_needle.sh

# Run RULER Evaluation
# 1. Prepare RULER data
cd benchmarks/RULER
./ruler_prepare_data.sh

# 2. Run Evaluation
cd ../..
./scripts/eval_ruler.sh
```

## Supported Features

### Text
- **Model Type**
  - qwen3 (`Qwen3-0.6B`, `Qwen3-8B`, `Qwen3-14B` ...)
- **KV Method**
  - H2O ([H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html))
  - StreamingLLM ([Efficient Streaming Language Models with Attention Sinks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html))
  - SnapKV ([SnapKV: LLM Knows What You are Looking for Before Generation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html))
  - GemFilter ([Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction](https://arxiv.org/abs/2409.17422))
  - FastKV ([FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](https://arxiv.org/abs/2502.01068))
- **Benchmark**
  - [LongBench](https://github.com/THUDM/LongBench)
  - [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
  - [RULER](https://github.com/NVIDIA/RULER/tree/main)

### Multimodal
- [ ] Coming soon

## TODO
### Model Type
- [ ] llama
- [ ] mistral

### KV Method
- [ ] PyramidInfer
- [ ] ASL

### Benchmark
- [ ] InfiniteBench