# Omni-KVCache-Factory

<p align="center">
  <img src="docs/images/logo.png" alt="Omni-KVCache-Factory Logo">
</p>

Omni-KVCache-Factory: A repository summarizing KVCache optimization methods, providing ready-to-use code for various methods, supporting multiple models and benchmarks to help researchers quickly run experiments and validate ideas.

<p align="center">📖 <a href="docs/README_zh.md">[简体中文] 🔧 <a href="docs/DevelopmentGuide.md">[开发指南]</a></p>

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