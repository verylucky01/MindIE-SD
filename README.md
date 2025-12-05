# README

## 简介

MindIE SD（Mind Inference Engine Stable Diffusion）是 MindIE 的视图生成推理模型套件，它的目标是为稳定扩散（Stable Diffusion, SD）系列大模型提供在昇腾硬件及其软件栈上的端到端推理解决方案。该软件系统内部集成了各功能模块，并对外提供统一的编程接口。

## Latest News

-   11/30/2025：MindIE SD正式宣布开源并面向公众开放！
-   11/30/2025: We are excited to announce that MindIE SD is now open source and available to the public!

## 架构介绍及关键特性
详见[架构介绍](docs/architecture_introduction.md)(包含：关键特性，目录设计等)

现支持主流扩散模型，对于部分diffusers模型进行昇腾亲和加速改造，模型归档在[Modelers](https://modelers.cn/models?name=MindIE&page=1&size=16)/[ModelZoo](https://www.hiascend.com/software/modelzoo)，模型列表详见[List of Supported Models](docs/architecture_introduction.md/table1198934616167)，也支持手动改造，详见examples。


## Getting started

本章节以Wan2.1模型为例，展示如何使用MindIE SD进行文生视频，关于该模型的更多推理内容请参见[链接](https://modelers.cn/models/MindIE/Wan2.1)。

1.  源码编译安装MindIE SD (镜像 / 软件包安装方式详见[developer_guide](docs/developer_guide.md))
    ```shell
    git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
    python setup.py bdist_wheel

    cd dist
    pip install mindie_sd-*.whl
    ```

2.  安装模型所需依赖并执行推理。

    在任意路径下载Wan2.1模型仓，并安装所需依赖，进行推理。用户可根据需要自行设置权重路径（例：/home/{用户名}/Wan2.1-T2V-14B）和推理脚本中的模型参数，参数解释详情请参见[参数配置](./examples/wan/parameter_config.md)。

    ```shell
    git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
    pip install -r requirements.txt

    # Wan2.1-T2V-14B 8卡推理
    bash examples/wan/infer_t2v.sh --model_base="/home/{用户名}/Wan2.1-T2V-14B"
    ```

## 加速特性效果

下面以Wan2.1模型为例，展示在Atlas 800I A2(1*64G)机器上单卡和多卡不同加速特性的加速效果。

其中cache表示使用[AttentionCache](./docs/features/kv_cache_offloading.md#attentioncache)特性, TP表示使用[Tensor Parallel](./docs/features/multi-device_parallelism.md#张量并行)特性, FA稀疏表示使用FA稀疏中的[RainFusion特性](./docs/features/lightweight_algorithm.md#fa稀疏)，CFG表示使用[CFG并行](./docs/features/multi-device_parallelism.md#cfg并行)特性，Ulysses表示使用[Ulysses并行](./docs/features/multi-device_parallelism.md#ulysses-sequence-parallel)加速特性，模型生成的视频的H\*W为832\*480, sample_steps为50。

### 单卡加速效果

#### cache加速效果

| baseline <br> 1024.8s| +高性能FA算子 <br> 766.6s 1.34x| + cache <br> 612.6s 1.67x|
|:---:|:---:|:---:|
|![](./docs/figures/t2v-14B_832_480_单卡base.gif)|![](./docs/figures/t2v-14B_832_480_单卡+FA.gif)|![](./docs/figures/t2v-14B_832_480_单卡%20+%20FA%20+%20attentioncache3.gif)|


#### 稀疏加速效果

| + FA稀疏度0.55 <br> 611.4s 1.68x| + FA稀疏度0.64 <br> 588.7s 1.74x| + FA稀疏度0.8 <br> ***560.5s** 1.83x|
|:---:|:---:|:---:|
|![](./docs/figures/t2v-14B_832_480_单卡%20+%20FA%20+%20attentioncache%20+%20FA%20稀疏%201.gif)|![](./docs/figures/t2v-14B_832_480_单卡%20+%20FA%20+%20attentioncache%20+%20FA%20稀疏%202.gif)|![](./docs/figures/t2v-14B_832_480_单卡%20+%20FA%20+%20attentioncache%20+%20FA%20稀疏3.gif)|


### 并行策略效果

#### 两卡单个并行策略效果
| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache算法优化| FA稀疏 | 50步E2E耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 2 | VAE | 832\*480 | √ | √ | √ | 548.8 | 1.02x|
| Wan2.1 | 2 | TP | 832\*480 | √ | √ | √ | 502.8 | 1.12x|
| Wan2.1 | 2 | CFG | 832\*480 | √ | √ | √ | 332.6 | 1.69x|
| Wan2.1 | 2 | Ulysses | 832\*480 | √ | √ | √ | 327.6 | ***1.71x**|

注：*号表示最优加速效果

#### 多卡并行策略组合效果

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache算法优化| FA稀疏 | 50步E2E耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 4 | TP=4,VAE | 832\*480 | √ | √ | √ | 204.0 | 2.754x|
| Wan2.1 | 4 | CFG=2,TP=2,VAE | 832\*480 | √ | √ | √ | 175.8 | 3.19x|
| Wan2.1 | 4 | Ulysses=4,VAE | 832\*480 | √ | √ | √ | 151.1 | 3.71x|
| Wan2.1 | 4 | CFG=2,Ulysses=2,VAE | 832\*480 | √ | √ | √ | 147.9 | ***3.79x**|
| Wan2.1 | 8 | TP=8,VAE | 832\*480 | √ | √ | √ | 141.5| 3.96x|
| Wan2.1 | 8 | CFG=2,TP=4,VAE | 832\*480 | √ | √ | √ | 102.9 | 5.45x|
| Wan2.1 | 8 | Ulysses=8,VAE | 832\*480 | √ | √ | √ | 78.1 | 7.18x|
| Wan2.1 | 8 | CFG=2,Ulysses=4,VAE | 832\*480 | √ | √ | √ | 76.4 | ***7.34x**|

注：*号表示最优加速效果
