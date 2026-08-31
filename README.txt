Python Coding Agent

使用 DeepSeek OpenAI 兼容接口的本地 Coding Agent，支持多步工具、多轮会话、SQLite、diff、连续回滚、审批和 DEBUG 日志。

WSL Ubuntu 22.04：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的 Key'
python -m coding_agent --session --workspace ./demo --debug

会话命令：

/new [标题]
/resume <ID>
/list
/diff [Turn 序号]
/rollback
/history [数量]
/delete
/help
/exit

每个 Turn 前后生成去重快照。/diff 彩色显示差异；/rollback 可连续恢复；/history 显示完整交流；/delete 经确认后删除会话。若工作区后来被外部修改，回滚会拒绝覆盖。

测试交互程序时使用 run_command 的 stdin，不开放 Shell 管道。当前 Turn 通过 write_file 新建的文件可自动删除；删除既有文件、移动或覆盖文件、执行检测到破坏行为的 Python 脚本需要用户批准。越界、受保护路径和内联代码始终禁止。

Token 优先采用 API usage，否则本地估算。上下文达到限制的 80% 时，LLM 增量摘要较老 Turn；原始历史和快照不删除，回滚摘要覆盖的 Turn 会使摘要失效。

CODING_AGENT_CONTEXT_TOKEN_LIMIT 调整限制（默认 64000）；CODING_AGENT_LLM_AUDIT=1 启用可选命令审计。本地硬策略始终优先。另支持 --max-steps、--max-tool-calls 和 CODING_AGENT_DEBUG。
