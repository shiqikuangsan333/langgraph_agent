# -*- coding: utf-8 -*-
"""
启动入口
========
用法：
    python main.py            # 默认运行 v2（多智能体 + RAG）
    python main.py --v1       # 运行第一代（单智能体 + 工具）

首次使用：把 .env.example 复制为 .env 并填入你的 API Key。
"""
import sys
from dotenv import load_dotenv
load_dotenv()  # 必须在导入 agent 模块之前加载 .env

import uvicorn

version = "v5"
if "--v1" in sys.argv:
    version = "v1"

if version == "v1":
    print("运行第一代: agent_v1_langgraph.py（单智能体 + 工具调用）")
    from agent_v1_langgraph import app
else:
    print("运行 v2: agent_v2_langgraph.py（多智能体评审 + RAG）")
    from agent_v2_langgraph import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
