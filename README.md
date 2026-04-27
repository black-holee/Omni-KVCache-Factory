# Omni-KVCache-Factory
Omni-KVCache-Factory: A repository summarizing KVCache optimization methods, providing ready-to-use code for various methods, supporting multiple models and benchmarks to help researchers quickly run experiments and validate ideas.

## Usage
### 1. Installation
Installation with the requirements package.
```
conda create -n okfc python=3.9
conda activate okfc
cd Omni-KVCache-Factory
pip install -r requirements.txt
pip install flash-attn==2.6.3
```

### 2. Quick Start
Inference with KVCache methods and evaluation for LongBench, Ruler, Needle-in-a-Haystack, and speedup benchmark.

```
# Run LongBench Evaluation
./scripts/eval_longbench.sh

# Run RULER Evaluation
./scripts/eval_ruler.sh

# Run Needle-in-a-Haystack Evaluation
./scripts/eval_needle.sh

# Run E2E Latency Benchmark
./scripts/eval_e2e.sh

# Run Prefill Latency Benchmark
./scripts/eval_prefill.sh
```