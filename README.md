# README

## 简介<a name="section6692115173811"></a>

MindIE SD（Mind Inference Engine Stable Diffusion）是 MindIE 的视图生成推理模型套件，它的目标是为稳定扩散（Stable Diffusion, SD）系列大模型提供在昇腾硬件及其软件栈上的端到端推理解决方案。该软件系统内部集成了各功能模块，并对外提供统一的编程接口。

## 会议日历<a name="section13231230133818"></a>

-   11/30/2025：MindIE SD正式宣布开源并面向公众开放！
-   11/30/2025: We are excited to announce that MindIE SD is now open source and available to the public!

## 架构介绍及关键特性<a name="section09781640191216"></a>
详见[架构介绍](docs/架构介绍.md)(包含：关键特性，目录设计等)

现支持主流扩散模型，对于部分diffusers模型进行昇腾亲和加速改造，归档在[Modelers](https://modelers.cn/models?name=MindIE&page=1&size=16)/[ModelZoo](https://www.hiascend.com/software/modelzoo)；也支持手动改造，详见example。

## Getting started<a name="section101707448417"></a>

本章节以Wan2.1模型为例，展示如何使用MindIE SD进行文生视频，关于该模型的更多推理内容请参见[链接](https://modelers.cn/models/MindIE/Wan2.1)。

1.  安装环境。
    1.  安装MindIE SD。

        详情请参见《MindIE安装指南》中的“安装MindIE\>[方式一：镜像部署方式](https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0021.html)”章节。

    2.  安装gcc、g++。

        若镜像环境中没有gcc、g++，请用户使用以下命令自行安装，并导入头文件路径

        ```
        yum install gcc g++ -y
        export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
        ```

    3.  安装模型所需依赖。

        使用以下命令在任意路径（例如：/home/_\{用户名\}_/code）下载Wan2.1模型仓，并安装所需依赖。

        ```
        git clone https://modelers.cn/MindIE/Wan2.1.git
        cd Wan2.1
        pip install -r requirements.txt
        ```

    4.  准备模型权重。

        模型权重详细信息如表格所示，用户需自行设置权重路径（例：/home/_\{用户名\}_/example/Wan2.1-T2V-14B）。

        **表 1**  模型权重列表

        <a name="table822517510017"></a>
        <table><thead align="left"><tr id="row42261751705"><th class="cellrowborder" valign="top" width="16.11%" id="mcps1.2.4.1.1"><p id="p13172172254"><a name="p13172172254"></a><a name="p13172172254"></a>模型</p>
        </th>
        <th class="cellrowborder" valign="top" width="34.02%" id="mcps1.2.4.1.2"><p id="p17172322511"><a name="p17172322511"></a><a name="p17172322511"></a>说明</p>
        </th>
        <th class="cellrowborder" valign="top" width="49.87%" id="mcps1.2.4.1.3"><p id="p15172102851"><a name="p15172102851"></a><a name="p15172102851"></a>权重</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row11263114101711"><td class="cellrowborder" valign="top" width="16.11%" headers="mcps1.2.4.1.1 "><p id="p526304101710"><a name="p526304101710"></a><a name="p526304101710"></a>Wan2.1-T2V-14B</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.02%" headers="mcps1.2.4.1.2 "><p id="p14263174141711"><a name="p14263174141711"></a><a name="p14263174141711"></a>文生视频模型</p>
        </td>
        <td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.2.4.1.3 "><p id="p2026319415173"><a name="p2026319415173"></a><a name="p2026319415173"></a>权重文件请单击<a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main" target="_blank" rel="noopener noreferrer">链接</a>获取。</p>
        </td>
        </tr>
        <tr id="row181291045151718"><td class="cellrowborder" valign="top" width="16.11%" headers="mcps1.2.4.1.1 "><p id="p8129145141713"><a name="p8129145141713"></a><a name="p8129145141713"></a>Wan2.1-I2V-14B-480P</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.02%" headers="mcps1.2.4.1.2 "><p id="p101291445171712"><a name="p101291445171712"></a><a name="p101291445171712"></a>图生视频模型</p>
        </td>
        <td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.2.4.1.3 "><p id="p6129144531718"><a name="p6129144531718"></a><a name="p6129144531718"></a>权重文件请单击<a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/tree/main" target="_blank" rel="noopener noreferrer">链接</a>获取。</p>
        </td>
        </tr>
        <tr id="row1623154911176"><td class="cellrowborder" valign="top" width="16.11%" headers="mcps1.2.4.1.1 "><p id="p4232104911715"><a name="p4232104911715"></a><a name="p4232104911715"></a>Wan2.1-I2V-14B-720P</p>
        </td>
        <td class="cellrowborder" valign="top" width="34.02%" headers="mcps1.2.4.1.2 "><p id="p1232204951711"><a name="p1232204951711"></a><a name="p1232204951711"></a>图生视频模型</p>
        </td>
        <td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.2.4.1.3 "><p id="p11232154961717"><a name="p11232154961717"></a><a name="p11232154961717"></a>权重文件请单击<a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main" target="_blank" rel="noopener noreferrer">链接</a>获取。</p>
        </td>
        </tr>
        </tbody>
        </table>

2.  初始化环境变量。

    MindIE安装路径下有环境变量设置脚本“set\_env.sh“，默认路径为/usr/local/Ascend/mindie，用户需根据实际安装路径进行环境变量配置。

    ```
    source /usr/local/Ascend/mindie/set_env.sh
    ```

3.  执行推理。

    请参考以下样例进行推理，参数解释详情请参见表格。

    -   Wan2.1-T2V-14B 8卡推理

        ```
        model_base="/home/{用户名}/example/Wan2.1-T2V-14B"
        torchrun --nproc_per_node=8 generate.py \
              --task t2v-14B \
              --size 1280*720 \
              --ckpt_dir ${model_base} \
              --dit_fsdp \
              --t5_fsdp \
              --sample_steps 50 \
              --ulysses_size 8 \
              --vae_parallel \
              --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
              --use_attentioncache \
              --start_step 20 \
              --attentioncache_interval 2 \
              --end_step 47
        ```

    -   Wan2.1-I2V-14B-480P 8卡推理

        ```
        model_base="/home/{用户名}/example/Wan2.1-I2V-14B-480P/"
        torchrun --nproc_per_node=8 generate.py \
              --task i2v-14B \
              --size 832*480 \
              --ckpt_dir ${model_base} \
              --frame_num 81 \
              --sample_steps 40 \
              --dit_fsdp \
              --t5_fsdp \
              --cfg_size 1 \
              --ulysses_size 8 \
              --vae_parallel \
              --image examples/i2v_input.JPG \
              --base_seed 0 \
              --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
              --use_attentioncache \
              --start_step 12 \
              --attentioncache_interval 4 \
              --end_step 37
        ```

    -   Wan2.1-I2V-14B-720P 8卡推理

        ```
        model_base="/home/{用户名}/example/Wan2.1-I2V-14B-720P/"
        torchrun --nproc_per_node=8 generate.py \
              --task i2v-14B \
              --size 1280*720 \
              --ckpt_dir ${model_base} \
              --frame_num 81 \
              --sample_steps 40 \
              --dit_fsdp \
              --t5_fsdp \
              --cfg_size 1 \
              --ulysses_size 8 \
              --vae_parallel \
              --image examples/i2v_input.JPG \
              --base_seed 0 \
              --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
              --use_attentioncache \
              --start_step 12 \
              --attentioncache_interval 4 \
              --end_step 37
        ```

        **表 2**  参数解释

        <a name="table8470029931"></a>
        <table><thead align="left"><tr id="row347116291633"><th class="cellrowborder" valign="top" width="21.060000000000002%" id="mcps1.2.4.1.1"><p id="p184601755194118"><a name="p184601755194118"></a><a name="p184601755194118"></a>参数名</p>
        </th>
        <th class="cellrowborder" valign="top" width="18.93%" id="mcps1.2.4.1.2"><p id="p7460155516416"><a name="p7460155516416"></a><a name="p7460155516416"></a>参数含义</p>
        </th>
        <th class="cellrowborder" valign="top" width="60.01%" id="mcps1.2.4.1.3"><p id="p84608550417"><a name="p84608550417"></a><a name="p84608550417"></a>取值</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row1147114291237"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p2037213644411"><a name="p2037213644411"></a><a name="p2037213644411"></a>model_base</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1637233617442"><a name="p1637233617442"></a><a name="p1637233617442"></a>权重路径</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p11372153624420"><a name="p11372153624420"></a><a name="p11372153624420"></a>模型权重所在路径。</p>
        </td>
        </tr>
        <tr id="row1392552918328"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p12925172953215"><a name="p12925172953215"></a><a name="p12925172953215"></a>task</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p12925182933218"><a name="p12925182933218"></a><a name="p12925182933218"></a>任务类型</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p1292502910324"><a name="p1292502910324"></a><a name="p1292502910324"></a>支持t2v-14B和i2v-14B。</p>
        </td>
        </tr>
        <tr id="row12468867107"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p194681468109"><a name="p194681468109"></a><a name="p194681468109"></a>size</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p64681068102"><a name="p64681068102"></a><a name="p64681068102"></a>视频分辨率</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p20345194662814"><a name="p20345194662814"></a><a name="p20345194662814"></a>生成视频的宽*高。</p>
        <a name="ul172121649202811"></a><a name="ul172121649202811"></a><ul id="ul172121649202811"><li>t2v-14B：模型默认值为1280*720；</li><li>i2v-14B-480P：模型默认值为[832, 480]、[720, 480]；</li><li>i2v-14B-720P：模型默认值为[1280, 720]。</li></ul>
        </td>
        </tr>
        <tr id="row4174145417181"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p8174195491814"><a name="p8174195491814"></a><a name="p8174195491814"></a>frame_num</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p41741154181816"><a name="p41741154181816"></a><a name="p41741154181816"></a>生成视频的帧数</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p17174185410180"><a name="p17174185410180"></a><a name="p17174185410180"></a>默认值为81帧。</p>
        </td>
        </tr>
        <tr id="row180313214350"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p6804721153516"><a name="p6804721153516"></a><a name="p6804721153516"></a>sample_steps</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p158042021163512"><a name="p158042021163512"></a><a name="p158042021163512"></a>采样步数</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p178041921173514"><a name="p178041921173514"></a><a name="p178041921173514"></a>扩散模型的迭代降噪步数，t2v模型默认值为50，i2v模型默认值为40。</p>
        </td>
        </tr>
        <tr id="row1235851163710"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p1535801143715"><a name="p1535801143715"></a><a name="p1535801143715"></a>prompt</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p11358214377"><a name="p11358214377"></a><a name="p11358214377"></a>文本提示词</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p12358181183714"><a name="p12358181183714"></a><a name="p12358181183714"></a>用户自定义，用于控制视频生成。</p>
        </td>
        </tr>
        <tr id="row1476210452117"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p57621342211"><a name="p57621342211"></a><a name="p57621342211"></a>image</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p147625412111"><a name="p147625412111"></a><a name="p147625412111"></a>用于生成视频的图片路径</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p11762748216"><a name="p11762748216"></a><a name="p11762748216"></a>i2v模型推理所需，用户自定义，用于控制视频生成。</p>
        </td>
        </tr>
        <tr id="row1046211199392"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p5462151973911"><a name="p5462151973911"></a><a name="p5462151973911"></a>base_seed</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p54621819193910"><a name="p54621819193910"></a><a name="p54621819193910"></a>随机种子</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p15462161912392"><a name="p15462161912392"></a><a name="p15462161912392"></a>用于视频生成的随机种子。</p>
        </td>
        </tr>
        <tr id="row1321483517395"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p22151835183910"><a name="p22151835183910"></a><a name="p22151835183910"></a>use_attentioncache</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1421543511397"><a name="p1421543511397"></a><a name="p1421543511397"></a>使能attentioncache算法优化</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p485895083013"><a name="p485895083013"></a><a name="p485895083013"></a>此优化为有损优化，如开启此优化，则需设置参数：start_step、attentioncache_interval、end_step。</p>
        <a name="ul12436145316300"></a><a name="ul12436145316300"></a><ul id="ul12436145316300"><li>start_step：cache开始的step；</li><li>attentioncache_interval：连续cache数；</li><li>end_step：cache结束的step。</li></ul>
        </td>
        </tr>
        <tr id="row185991037277"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p76004312711"><a name="p76004312711"></a><a name="p76004312711"></a>nproc_per_node</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1460011372711"><a name="p1460011372711"></a><a name="p1460011372711"></a>并行卡数</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><a name="ul6979743282"></a><a name="ul6979743282"></a><ul id="ul6979743282"><li>Wan2.1-T2V-14B支持的卡数为1、2、4或8。</li><li>Wan2.1-I2V-14B支持的卡数为1、2、4或8。</li></ul>
        </td>
        </tr>
        <tr id="row16261195693912"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p2261155643910"><a name="p2261155643910"></a><a name="p2261155643910"></a>ulysses_size</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p13261256153911"><a name="p13261256153911"></a><a name="p13261256153911"></a>ulysses并行数</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p1526135612397"><a name="p1526135612397"></a><a name="p1526135612397"></a>默认值为1，ulysses_size * cfg_size = nproc_per_node。</p>
        </td>
        </tr>
        <tr id="row111392315243"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p1711482312419"><a name="p1711482312419"></a><a name="p1711482312419"></a>cfg_size</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p13114162312249"><a name="p13114162312249"></a><a name="p13114162312249"></a>cfg并行数</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p01141523162419"><a name="p01141523162419"></a><a name="p01141523162419"></a>默认值为1，ulysses_size * cfg_size = nproc_per_node。</p>
        </td>
        </tr>
        <tr id="row1259012559561"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p1359055518568"><a name="p1359055518568"></a><a name="p1359055518568"></a>dit_fsdp</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1759010553565"><a name="p1759010553565"></a><a name="p1759010553565"></a>DiT使用FSDP</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p12590055185611"><a name="p12590055185611"></a><a name="p12590055185611"></a>DiT模型是否使用完全分片数据并行（Fully Sharded Data Parallel, FSDP）策略。</p>
        </td>
        </tr>
        <tr id="row431618018575"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p153177019575"><a name="p153177019575"></a><a name="p153177019575"></a>t5_fsdp</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p33174018573"><a name="p33174018573"></a><a name="p33174018573"></a>T5使用FSDP</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p9317301573"><a name="p9317301573"></a><a name="p9317301573"></a>文本到文本传输转换（Text-To-Text Transfer Transformer, T5）模型是否使用FSDP策略。</p>
        </td>
        </tr>
        <tr id="row11402154312018"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p194039438019"><a name="p194039438019"></a><a name="p194039438019"></a>vae_parallel:</p>
        </td>
        <td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p24036431804"><a name="p24036431804"></a><a name="p24036431804"></a>使能vae并行策略</p>
        </td>
        <td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p1940334314013"><a name="p1940334314013"></a><a name="p1940334314013"></a>vae模型是否使用并行策略。</p>
        </td>
        </tr>
        </tbody>
        </table>

