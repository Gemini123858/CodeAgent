Python Coding Agent

项目目标是自行实现一个轻量级本地 Coding Agent，不使用任何 Agent 框架。

当前进度（第一阶段）：通过 DeepSeek 的 OpenAI 兼容接口完成一次普通对话请求，用于验证环境变量、SDK、网络和模型配置。工具调用、工作区操作和 Agent 循环将在后续阶段逐步实现。

WSL Ubuntu 22.04 启动：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的 Key'
python -m coding_agent "请用一句话介绍你自己"

也可以省略消息参数，启动后再输入：

python -m coding_agent

Git 仓库地址：待补充
