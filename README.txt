Python Coding Agent

一个不依赖 Agent 框架、使用 DeepSeek OpenAI 兼容接口的本地 Coding Agent。支持多步工具循环、多轮会话、SQLite 持久化、工作区 diff、连续单步回滚、命令审批和 DEBUG 日志。

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

每个 Turn 前后都会生成内容寻址快照，文件内容保存在 .coding-agent/snapshots 的去重 blob 仓库。/diff 彩色展示最近未回滚 Turn 的差异，也可查看指定 Turn；/rollback 恢复最近 Turn，可连续执行。/history 查看当前会话的交流和工具摘要；/delete 经确认后删除当前会话的数据库记录及独占快照。若工作区在 Turn 后又被外部修改，回滚会拒绝覆盖。

测试交互程序时使用 run_command 的 stdin，不开放 Shell 管道。当前 Turn 通过 write_file 新建的文件可自动删除；删除既有文件、移动或覆盖文件、执行检测到破坏行为的 Python 脚本需要用户批准。越界、受保护路径和内联代码始终禁止。

SQLite 同时保存消息、工具结果、文件变化、审批、快照和回滚链。静态脚本检查及文件级快照不是操作系统沙箱。

可用 --max-steps、--max-tool-calls，或 CODING_AGENT_MAX_STEPS、CODING_AGENT_MAX_TOOL_CALLS、CODING_AGENT_DEBUG 调整运行参数。
