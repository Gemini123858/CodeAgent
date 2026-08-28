Python Coding Agent

项目目标是自行实现一个轻量级本地 Coding Agent，不使用任何 Agent 框架。

当前进度：已完成 DeepSeek 对话、本地工具和单次任务 Agent 循环。模型可反复调用 list_files、read_file、write_file、run_command，程序执行并回传结果，直到模型输出最终答案。当前上下文策略 FullHistoryContext 会保留并重发完整历史，后续可替换为截断或摘要策略。

WSL Ubuntu 22.04 启动：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的 Key'
python -m coding_agent "请用一句话介绍你自己"

独立测试本地工具：

mkdir -p demo
python -m coding_agent.tools --workspace ./demo write-file hello.py --content "print('hello')"
python -m coding_agent.tools --workspace ./demo list-files

测试单次任务闭环：

python -m coding_agent --agent --workspace ./demo --max-steps 12 --debug "创建 hello.py 并运行"

也可设置 CODING_AGENT_DEBUG=1 和 CODING_AGENT_MAX_STEPS=12。调试输出每步请求、响应和工具结果，并隐藏 API Key 与 reasoning_content。当前关闭 DeepSeek 思考模式。

文件工具限制在指定工作区并保护 .git、.env 等路径；命令工具不启用 Shell、限制命令并移除敏感环境变量，但仍不是操作系统级沙箱。

Git 仓库地址：待补充
