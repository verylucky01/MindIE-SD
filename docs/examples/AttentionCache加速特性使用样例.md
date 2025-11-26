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

(1) 以多卡推理为例，通过搜索工具计算出最佳配置参数后，修改`run_edit_2509_cfg_usp.py`文件的`main()`方法中关于cache的代码，关键代码如下所示：
```python
    if COND_CACHE or UNCOND_CACHE:
        # Attention cache
        cache_config = CacheConfig(
            method="attention_cache",
            blocks_count=len(pipeline.transformer.transformer_blocks),  # 使能cache的block的个数
            steps_count=args.num_inference_steps,                       # 模型推理的总迭代步数
            step_start=12,                                              # 开始进行cache的步数索引
            step_interval=5,                                            # 强制重新计算的间隔步数
            step_end=37,                                                # 停止cache的步数索引
        )
        pipeline.transformer.cache_cond = CacheAgent(cache_config) if COND_CACHE else None      # 正样本配置缓存
        pipeline.transformer.cache_uncond = CacheAgent(cache_config) if UNCOND_CACHE else None  # 负样本配置缓存
        print("启用缓存配置")
```
(2) 可以通过配置环境变量`开启AttentionCache`
```shell
# 启用正样本cache
export COND_CACHE=1

# 启用负样本cache
export UNCOND_CACHE=1
```

(3) 执行以下命令开启cache优化并进行推理，可通过对比开启cache前后的模型平均推理时间观察加速效果。
```shell
export LCCL_DETERMINISTIC=true  
export HCCL_DETERMINISTIC=true  
export ATB_MATMUL_SHUFFLE_K_ENABLE=0  
export ATB_LLM_LCOC_ENABLE=true      
export CLOSE_MATMUL_K_SHIFT=true     

model_path="/mnt/data/Qwen-Image-Edit-2509"

# 开启cache算法优化
export COND_CACHE=1
export UNCOND_CACHE=1

# 8卡并行，cfg_size * ulysses_size = 8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
torchrun --nproc_per_node=8 --master-port 29508 run_edit_2509_cfg_usp.py \
    --model_path ${model_path} \
    --num_inference_steps 50 \
    --seed 42 \
    --output_dir "attention_cache_baseline_cfg2_ulysses4_optimize_1+2_50steps" \
    --ulysses_size 4 \
    --cfg_size 2 \
--img_paths ./yarn-art-pikachu.png
```
参数说明：
- ASCEND_RT_VISIBLE_DEVICES: 执行模型推理的设备id，对于16卡机器，需要设定为连续的前8张或后8张
- model_path: 模型权重路径
- num_inference_steps: 推理的步数
- seed: 设定种子
- output_dir: 保存推理结果的路径
- ulysses_size: ulysses并行数，使用时设定为24的因数
- cfg_size: cfg并行数，使用时只能设定为2
- img_paths: 输入图片路径，多图则用逗号分隔，如`img1,img2`

---
# 3. FAQ
> Q：Qwen-Image-Edit-2509在开启AttentionCache后推理报错：RuntimeError: NPU out of memory. 

A：开启AttentionCache会增加对显存的消耗，单卡显存容易不足，推荐使用多卡推理。
