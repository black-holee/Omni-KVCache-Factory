# Development Guide
## 1. 扩展模型
### 1.1 找到模型源代码
transformers库中的各个模型源代码位于‘miniconda3/envs/okfc/lib/python3.10/site-packages/transformers/models’目录下，以qwen3为例：

找到“miniconda3/envs/okfc/lib/python3.10/site-packages/transformers/models/qwen3/modeling_qwen3.py”文件，这个文件中包含了qwen3的模型定义。

我们只需要找到其中的Attention、DecoderLayer和Model类，对于qwen3来说，则为：Qwen3Attention、Qwen3DecoderLayer和Qwen3Model。

### 1.2 对源码进行修改
**注意：不要直接对库中的源码进行修改。**

参考baselines目录下的代码，将所需的代码复制到“baselines/{method}/{model_type}_model.py”中，要根据扩展模型的model_type修改文件的名称，然后参考现有代码的修改方式，对复制来的源码进行修改。

baselines中的KV Cache优化方法分为两类，一类例如fastkv，会对token进行裁剪，另一类如snapkv，全程只对kv cache进行裁剪。

- 对于只裁剪kv cache的方法，通常只需要修改Attention.forward方法，具体可以参考”baselines/snapkv/qwen3_model.py“。

- 对于会裁剪token的方法，通常需要修改Attention类、DecoderLayer.forward和Model.forward方法，具体可以参考”baselines/fastkv/qwen3_model.py“。

### 1.3 在代码执行时替换源码
将修改后的方法添加到”baselines/monkeypatch.py“文件中，具体可以参考”baselines/monkeypatch.py“，在执行测试时运行”replace_{model_type}“方法即可完成对源码的替换，可以参考”infer_eval/run_longbench.py“。