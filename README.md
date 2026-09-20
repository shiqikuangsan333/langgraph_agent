# LangGraph 可定制性格智能体 · v3 工程版

## 项目结构（仿 paper_agent 规范布局）

```
langgraph_agent_v3/
├── main.py                  # 启动入口：python main.py（默认 v5，--v1 切基础版）
├── agent_v1_langgraph.py    # 第一代：单智能体 + 工具调用
├── agent_v2_langgraph.py    # 第二代：多智能体评审 + RAG（当前版本）
├── langgraph_chat.html      # 网页前端（人设工坊 + Markdown/公式渲染）
├── .env                     # 你的 API 配置（已被 git 忽略，别外传）
├── .env.example             # 配置模板
├── requirements.txt         # 依赖清单
├── .gitignore
├── knowledge/               # RAG 知识库：放 .txt/.md/.pdf 课件
│   └── 通信原理笔记.md
├── reports/                 # 运行报告/对话记录（自己往里加）
└── 面试复盘.md               # 简历描述 + 10 个高频面试问答
```

## 快速开始

```bash
pip install -r requirements.txt     # 装依赖

cp .env.example .env                # Windows: copy .env.example .env
# 编辑 .env 填入你的 API_KEY

python main.py                      # 启动 v5 最终版
# 或 python main.py --v1            # 体验基础版
```

浏览器打开 http://localhost:8000

## 版本演进路线（面试可讲）

| 版本 | 文件 | 能力 |
|------|------|------|
| 第一代 | agent_v1_langgraph.py | 单智能体 + 3 个工具（计算/知识/时间），状态图两节点 |
| 第二代 | agent_v2_langgraph.py | 多智能体（生成+评审双节点）、RAG 知识库、.env 配置 |

第二代之后的迭代（v3 流式输出、v4 长期记忆等）可以自己练手补，
就像你的 paper_agent 从 agent.py 迭代到 agent_v5_langgraph.py 一样（那边迭代到 v5，咱们一步一个脚印）。

## 体验建议

1. `python main.py --v1` 先跑第一代，发"帮我算 log2(1+15)"
2. Ctrl+C 停掉，`python main.py` 跑 v5，问"奈奎斯特定理是什么"（走 RAG）
3. 对比两版差异，配合 面试复盘.md 里的问答理解为什么这样设计
