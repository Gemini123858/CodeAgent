Python Coding Agent

一个不依赖 Agent 框架的本地 Coding Agent，使用 DeepSeek 的 OpenAI 兼容接口。当前支持多步工具循环、多轮会话、SQLite 持久化、DEBUG 日志，以及 list_files、read_file、write_file、run_command 四个工作区工具。

WSL Ubuntu 22.04：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的 Key'

启动多轮会话：

python -m coding_agent --session --workspace ./demo --debug

会话命令：/new [标题]、/resume <ID>、/list、/help、/exit。会话数据保存在工作区的 .coding-agent/state.db，该目录不会暴露给模型工具。

执行一次持久化任务：

python -m coding_agent --agent --workspace ./demo "创建 hello.py 并运行"

可用 --max-steps 和 --max-tool-calls 设置单轮上限，也可设置 CODING_AGENT_MAX_STEPS、CODING_AGENT_MAX_TOOL_CALLS、CODING_AGENT_DEBUG。

每次请求由 RequestBuilder 重新构造完整历史和工具；当前传入全部工具，已预留 EmbeddingProvider/EmbeddingToolSelector。工具失败会作为 error 工具消息保留，让模型下一步修正；调用超限时整批跳过并补齐对应结果；API 失败和步骤超限均记录 Turn 状态。回滚尚未实现，因此不会删除可能已产生副作用的失败轮次。

工具只允许访问指定工作区并保护 .git、.env、.coding-agent；命令执行不启用 Shell、限制命令并移除敏感环境变量，但不等同于操作系统级沙箱。
