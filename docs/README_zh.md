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
