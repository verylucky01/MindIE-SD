> 以`Qwen-Image-Edit-2509`模型为例，展示如何使用`DiTCache`加速特性，进行模型优化

# 1. 前置准备

(1) 权重下载
- 原始权重来自[HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- 国内权重可使用[此处](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509)

(2) 使用以下命令在任意路径（例如：/home/{用户名}/example/）下载模型代码并安装所需依赖
```shell
git clone https://modelers.cn/MindIE/Qwen-Image-Edit-2509.git && cd Qwen-Image-Edit-2509
pip install -r requirements.txt
```
(3) 将`examples/cache`目录下的[cache.py](cache.py)文件拷贝到Qwen-Image-Edit-2509目录下

关于该模型的更多内容请参考[魔乐社区](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)。

# 2. 使能DiTCache

执行以下命令开启cache优化并进行推理，通过对比开启cache前后的模型平均推理时间观察加速效果。
```shell
export COND_CACHE=1
export UNCOND_CACHE=1

python cache.py  \
--model_path /mnt/data/Qwen-Image-Edit-2509  \
--device_id 0  \
--img_paths ./yarn-art-pikachu.png
```
参数说明：
- model_path: 模型权重路径
- device_id: 执行模型推理的设备id
- img_paths: 输入图片路径，多图则用逗号分隔，如`img1,img2`

注：若关闭cache加速，将环境变量`COND_CACHE`和`UNCOND_CACHE`赋值为0