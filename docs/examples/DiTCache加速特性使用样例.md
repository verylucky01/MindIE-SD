> 以`Qwen-Image-Edit-2509`模型为例，展示如何使用`DiTCache`加速特性，进行模型优化指导。

# 1. 前置准备

(1) 使用以下命令在任意路径（例如：/home/{用户名}/example/）下载模型代码
```shell
git clone https://modelers.cn/MindIE/Qwen-Image-Edit-2509.git
```

(2) 使用以下命令进入Qwen-Image-Edit-2509文件夹并安装所需依赖
```shell
cd Qwen-Image-Edit-2509
pip install diffusers==0.35.1
pip install transformers==4.52.4
pip install yunchang==0.6.0
```

(3) 准备模型权重
```shell
https://huggingface.co/Qwen/Qwen-Image-Edit-2509
```

关于该模型的更多内容请参考[魔乐社区](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)。

# 2. 优化示例

(1) 以单卡推理为例，通过搜索工具计算出最佳配置参数后，修改`run_edit_2509.py`文件的`main()`方法中关于cache的代码，关键代码如下所示：
```python
    if COND_CACHE or UNCOND_CACHE:
        # DiT cache
        cache_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=60,                        # 使能cache的block的个数
            steps_count=args.num_inference_steps,   # 模型推理的总迭代步数
            step_start=10,                          # 开始进行cache的步数索引
            step_interval=3,                        # 强制重新计算的间隔步数
            step_end=35,                            # 停止cache的步数索引
            block_start=10,                         # 每一步中，开始进行cache的block索引
            block_end=50                            # 每一步中，停止cache的block索引
        )
        pipeline.transformer.cache_cond = CacheAgent(cache_config) if COND_CACHE else None      # 正样本配置缓存
        pipeline.transformer.cache_uncond = CacheAgent(cache_config) if UNCOND_CACHE else None  # 负样本配置缓存
        print("启用缓存配置")
```

(2) 可以通过配置环境变量`开启DiTCache`
```shell
# 启用正样本cache
export COND_CACHE=1

# 启用负样本cache
export UNCOND_CACHE=1
```

(3) 执行以下命令开启cache优化并进行推理，可通过对比开启cache前后的模型平均推理时间观察加速效果。
```shell
# 开启cache算法优化
export COND_CACHE=1
export UNCOND_CACHE=1

# 单卡执行
python run_edit_2509.py  \
--model_path /mnt/data/Qwen-Image-Edit-2509  \
--device_id 0  \
--img_paths ./yarn-art-pikachu.png
```
参数说明：
- model_path: 模型权重路径
- device_id: 执行模型推理的设备id
- img_paths: 输入图片路径，多图则用逗号分隔，如`img1,img2`
