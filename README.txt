Python Coding Agent

项目目标是自行实现一个轻量级本地 Coding Agent，不使用任何 Agent 框架。

当前进度：第一阶段已完成 DeepSeek 单轮对话请求；第二阶段已独立实现 list_files、read_file、write_file 和 run_command 四个本地工具，以及工作区路径校验、文件大小限制、命令白名单和超时控制。工具暂未接入模型，Agent 循环将在后续阶段实现。

WSL Ubuntu 22.04 启动：

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY='你的 Key'
python -m coding_agent "请用一句话介绍你自己"

也可以省略消息参数，启动后再输入：

python -m coding_agent

独立测试本地工具：

mkdir -p demo
python -m coding_agent.tools --workspace ./demo write-file hello.py --content "print('hello')"
python -m coding_agent.tools --workspace ./demo list-files
python -m coding_agent.tools --workspace ./demo read-file hello.py
python -m coding_agent.tools --workspace ./demo run-command -- python3 hello.py

文件工具只能操作指定工作区并禁止直接访问 .git。命令工具不启用 Shell，只允许有限命令，并会移除传给子进程的疑似密钥环境变量；但它仍不是操作系统级沙箱，应仅对可信工作目录使用。

Git 仓库地址：待补充
