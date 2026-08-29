Python Coding Agent

一个不依赖 Agent 框架、使用 DeepSeek OpenAI 兼容接口的本地 Coding Agent。支持多步工具循环、多轮会话、SQLite 持久化、DEBUG 日志，以及 list_files、read_file、write_file、delete_file、run_command。

WSL Ubuntu 22.04：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的 Key'

启动：

python -m coding_agent --session --workspace ./demo --debug

会话命令：/new [标题]、/resume <ID>、/list、/help、/exit。数据保存在工作区 .coding-agent/state.db，该目录不会暴露给模型工具。

执行单轮任务：

python -m coding_agent --agent --workspace ./demo "创建 hello.py 并运行"

测试交互程序时，模型应通过 run_command 的 stdin 字段传入输入，不需要 Shell 管道。当前 Turn 通过 write_file 新建的普通文件可由 delete_file 自动删除；删除既有文件、移动或覆盖文件、执行检测到破坏行为的 Python 脚本时，CLI 会请求用户批准。拒绝结果会返回模型且相同请求不重复询问。越界、受保护路径、Shell 操作符和内联代码始终禁止。

SQLite 记录消息、工具结果、耗时、文件变化和审批决定，已为 diff、回滚及 LLM 命令审计预留接口。静态脚本检查不是操作系统沙箱，不能识别所有间接或混淆行为。

可用 --max-steps、--max-tool-calls，或 CODING_AGENT_MAX_STEPS、CODING_AGENT_MAX_TOOL_CALLS、CODING_AGENT_DEBUG 调整运行参数。
