# 快速开始

本章节以**Wan2.1**模型为例，展示如何使用MindIE SD进行文本生成视频，关于该模型的更多推理内容请参见 [Modelers - MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1)。

## 安装环境

源码编译安装MindIE SD（镜像 / 软件包安装方式详见 [developer_guide](./installing_guide.md)）

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
python setup.py bdist_wheel 
 
cd dist 
pip install mindiesd-*.whl 
```
  
若torch版本为2.6.0，则可直接pip安装MindIE SD，无需源码编译。

```bash
pip install --trusted-host ascend.devcloud.huaweicloud.com -i https://ascend.devcloud.huaweicloud.com/pypi/simple/ mindiesd
```

## 执行推理

安装模型所需依赖并执行推理。

在任意路径下载Wan2.1模型仓，并安装所需依赖。在MindIE SD代码路径下进行推理。用户可根据需要自行设置权重路径（例：/home/{用户名}/Wan2.1-T2V-14B）和推理脚本中的模型参数，参数解释详情请参见[参数配置](../../examples/wan/parameter_config.md)。

```bash
git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
pip install -r requirements.txt

# Wan2.1-T2V-14B 8 卡推理
cp MindIE-SD/examples/wan/infer_t2v.sh ./
bash infer_t2v.sh --model_base="/home/{用户名}/Wan2.1-T2V-14B"
```

## 加速特性效果展示

下面以Wan2.1模型为例，展示在Atlas 800I A2 推理服务器(1*64G) 上单卡和多卡实现不同加速特性的加速效果。

其中：

- Cache：表示使用[AttentionCache](./features/cache.md#attentioncache)特性；
- TP：表示使用[Tensor Parallel](./features/parallelism.md#张量并行)特性；
- FA稀疏：表示使用FA稀疏中的[RainFusion特性](./features/sparse_quantization.md#fa稀疏)；
- CFG：表示使用[CFG并行](./features/parallelism.md#cfg并行)特性；
- Ulysses：表示使用[Ulysses并行](./features/parallelism.md#ulysses-sequence-parallel)加速特性，模型生成的视频的H*W为832\*480，sample_steps为50。

### 单卡加速效果

**Cache 加速效果**

| Baseline | + Cache 加速比1.6 | + Cache 加速比2.0 | + Cache 加速比2.4 |
|:---:|:---:|:---:|:---:|
| 860.2s | 631.7s 1.36x | 541.8s 1.59x | 516.9s ***1.66x** |
| ![](./figures/单卡base%20+%20高性能FA算子.gif) | ![](./figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为1.6.gif) | ![](./figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为2.0.gif) | ![](./figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为2.4.gif) |

### 并行策略效果

**双卡单个并行策略效果**

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache 算法优化| FA 稀疏 | 50 步 E2E 耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 2 | VAE | 832*480 | √ | √ | √ | 548.8 | 1.02x|
| Wan2.1 | 2 | TP | 832*480 | √ | √ | √ | 502.8 | 1.12x|
| Wan2.1 | 2 | CFG | 832*480 | √ | √ | √ | 332.6 | 1.69x|
| Wan2.1 | 2 | Ulysses | 832*480 | √ | √ | √ | 327.6 | ***1.71x**|

注：* 表示最优加速效果。

**多卡并行策略组合效果**

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache 算法优化| FA 稀疏 | 50 步 E2E 耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 4 | TP=4, VAE | 832*480 | √ | √ | √ | 204.0 | 2.754x|
| Wan2.1 | 4 | CFG=2, TP=2, VAE | 832*480 | √ | √ | √ | 175.8 | 3.19x|
| Wan2.1 | 4 | Ulysses=4, VAE | 832*480 | √ | √ | √ | 151.1 | 3.71x|
| Wan2.1 | 4 | CFG=2, Ulysses=2, VAE | 832*480 | √ | √ | √ | 147.9 | ***3.79x**|
| Wan2.1 | 8 | TP=8, VAE | 832*480 | √ | √ | √ | 141.5| 3.96x|
| Wan2.1 | 8 | CFG=2, TP=4, VAE | 832*480 | √ | √ | √ | 102.9 | 5.45x|
| Wan2.1 | 8 | Ulysses=8, VAE | 832*480 | √ | √ | √ | 78.1 | 7.18x|
| Wan2.1 | 8 | CFG=2, Ulysses=4, VAE | 832*480 | √ | √ | √ | 76.4 | ***7.34x**|

注：* 表示最优加速效果。
