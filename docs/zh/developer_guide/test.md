# 测试

## 自测方法

本章节介绍MindIE SD仓算子自测方式，看护算子精度。

1. 使用以下命令卸载MindIE SD

    ```bash
    pip uninstall mindiesd
    ```

2. 运行全量UT测试用例。
    
    ```bash
    pip install -r MindIE-SD/requirements.txt
    pip install coverage
    cd MindIE-SD/tests
    bash run_test.sh
    ```

3. 运行LA单算子精度测试用例。

    修改MindIE-SD/tests/plugin/la_acc_prof.py文件，选择Option 1或Option 2，通过加载test_la.csv或enumerated_cases.csv文件，测试LA算子在所设置shape下的精度。

   - "./tests/plugin/test_la.csv"：设置了常用SD模型的输入shape。
   - "enumerated_cases.csv"：枚举的各种shape。

    完成修改后执行以下命令：
    
    ```bash
    cd MindIE-SD/tests
    python plugin/la_acc_prof.py
    ```

    运行成功后会在MindIE-SD目录下保存acc_output_results_1.csv和acc_output_results_1.csv两个结果文件，记录了la和fascore的相似度，可查看算子在所需shape下的精度。

## 问题定位

1. 常见异常说明。

    在使用MindIE SD进行推理业务时，模型相关文件（权重、配置、模型代码等）的安全性需由用户保证，常见的异常如下：
    - 如用户在模型初始化的时候，用户修改模型配置的默认参数可能会影响模型接口，若权重文件过大或配置文件中参数过大，可能会导致由out of memory导致的报错信息，例如："RuntimeError: NPU out of memory. Tried to allocate xxx GiB."。
    - 使用模型推理时，模型内部会进行推理计算，若用户输入的tensor shape过大，可能会出现由out of memory导致的报错信息，例如："RuntimeError: NPU out of memory. Tried to allocate xxx GiB."。
    - 在使用MindIE SD进行生成业务时，如果出现错误输入、环境不适配等问题时，代码中会抛出异常，需要用户在上层进行异常捕获处理，常见的异常类型见表格：

        |异常类型|说明|
        |--|--|
        |ZeroDivisionError|除0异常。|
        |ValueError|参数值异常。|
