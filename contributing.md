# contributing

## 提交Issue/处理Issue任务

MindIE-SD 社区基于 GitCode 提供的Issue管理功能，完整记录每一个开发者 Issue 的处理全流程，您可以对 Issue 进行查找、创建、评论、处理等操作，详细的操作步骤和说明如下：

1. 查找 Issue

在 MindIE-SD 社区首页选择 Issues 板块，在搜索框中输入关键字，即可对社区内所有项目的 Issue 标题及内容进行全局搜索。

2. 创建 Issue

如果您需要上报 Bug、提交需求，或为社区提供建议，请按以下步骤操作：

1）进入社区首页，在 Issue 板块中，单击 新建 Issue，即可进入Issue类型选择界面。

2）根据需求选择 Issue 类型（如 Bug-Report、新需求、空白Issue等），单击 使用该模板 或  创建 Issue 进入Issue详细页面。

3）在 Issue 填写界面中按提示填写相关信息：

-   标题：简要描述需求或问题要点。
-   内容：根据系统提供的模板提示详细填写具体内容，以便我们更好地理解和处理。

4）填写完成后，单击 提交 Issue 或 新建 Issue 完成创建。

3. 评论 Issue

每个 Issue 下方均支持开发者交流讨论，欢迎在评论区发表意见。

4. 处理 Issue

-   在评论框中输入 /assign （分配给自己）或 /assign @gitcode\_id（分配给特定账号），机器人会按assign 要求进行分配。
-   对应gitcode\_id 将显示在 Issue 负责人列表中。

前置条件：Issue 负责人必须是 Issue 提交者或项目成员。若希望成为项目成员，请联系[对应仓库 SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs)  maintainer 添加。


## 提交PR

为参与 MindIE-SD 社区贡献，提交 PR 前需完成开发环境准备，并仔细了解项目特定的开发规范和版权声明要求，确保您的贡献符合项目标准后再进入提交流程。详细的操作步骤和说明如下：

1.准备开发环境

如果您希望参与MindIE SD项目贡献（如代码、文档等），需要准备开发环境，请参考3.1构建指导，了解环境配置的具体要求。

2.版权声明

在参与项目贡献前，请务必仔细阅读 LICENSE 文件，并确保您的所有贡献符合该许可证的要求。

1）版权声明要求：

请在所有新建的源代码文件（如 .cpp, .h, .py 等）头部添加规范的版权声明。

2）声明模板：

请根据项目采用的许可证，选择对应的声明模板。

-   对于 Apache 2.0、MIT 等常见开源协议：建议从官方渠道获取标准的版权声明头。你可以访问  [Open Source Initiative](https://opensource.org/licenses)  或该许可证的官方网站查询具体要求。

-   对于 MindIE SD：请参考4.1 License中的声明文本

3.贡献提交流程

1）Fork仓库

-   将代码仓Fork到您的个人账户
-   克隆个人仓库到本地环境
-   在本地分支进行代码、文档等修改

2）本地验证

-   参考3.1章节进行本地构建与验证
-   确保代码符合贡献要求

3）提交Pull-Request

-   代码验证通过后，提交PR到MindIE SD开源项目
-   参照[社区评论命令](https://gitcode.com/cann/community/blob/master/docs/robot/cann/robot-command.md)中的对应命令触发门禁测试

4）代码审查

-   门禁测试未通过：根据门禁反馈修改代码
-   测试通过：PR 将分配给 Committer 进行审查，您可以在PR评论区通过@committer\_gitcode\_id提醒 Committer 进行审查，然后及时关注审查意见并进行相应调整。

5）代码合入

-   PR 审查通过后，代码将合入MindIE SD开源项目。
