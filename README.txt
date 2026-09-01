Python Coding Agent

Git 仓库地址

https://github.com/Gemini123858/CodeAgent

项目简介

这是一个使用 Python 从零实现的本地 Coding Agent，通过 DeepSeek OpenAI 兼容接口调用模型，不依赖 Agent 框架。它可在指定工作区内操作文件、执行受限命令，并通过多步工具循环完成编程任务。

如何运行

推荐环境：Ubuntu 22.04、Python 3.10 及以上。

git clone https://github.com/Gemini123858/CodeAgent.git
cd CodeAgent
python3 -m venv .venv
source .venv/bin/activate
pip install "openai>=1.0.0,<3"
mkdir -p demo
export DEEPSEEK_API_KEY='你的 API Key'
python -m coding_agent --session --workspace ./demo

--debug 可输出脱敏调试信息。会话支持 /new、/resume、/history、/context、/diff、/rollback 和 /delete 等命令。

特色功能

- 多步 Agent 循环：动态构造 messages 和 tools，支持失败反馈、步骤及工具调用上限。
- SQLite 持久化：保存会话、消息、模型请求、工具链、耗时和审批记录。
- 安全工具系统：限制工作区和命令；危险操作由本地规则拦截，中风险操作需用户确认，可选 LLM 审计。
- 可解释修改链：Turn 前后生成去重快照，彩色展示 diff，支持冲突检测和连续单步回滚。
- 上下文管理：优先使用 API token usage，否则本地估算；接近限制时增量摘要较早 Turn，原始历史仍完整保留。

其它说明

CODING_AGENT_CONTEXT_TOKEN_LIMIT 可调整上下文限制；CODING_AGENT_LLM_AUDIT=1 启用命令审计。命令白名单、路径检查和快照属于应用级保护，不能替代操作系统沙箱。
