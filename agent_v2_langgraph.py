# -*- coding: utf-8 -*-
"""
agent_v2_langgraph.py —— v2 智能体（多智能体评审 + RAG）
===========================================================
新特性：
  1. 多智能体：回复先经【评审节点】质检，不合格自动打回重写（最多2轮）
  2. RAG：把课件/笔记放进 knowledge 文件夹，智能体自动检索回答
  3. 工具调用：计算、时间、知识库检索

配置写在 .env 文件（复制 .env.example 改名即可），不要直接改本文件。
运行：python main.py（默认运行本版）；也可 python agent_v2_langgraph.py
"""
import os
import operator
from datetime import datetime
from typing import TypedDict, List, Any
from typing_extensions import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# =====================================================================
# 1. 修改这里：填你自己的 API 信息（支持任何 OpenAI 兼容接口）
# =====================================================================
# 从 .env 读取配置（由 main.py 的 load_dotenv() 加载；单独运行时在此手动加载）
from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
API_KEY  = os.getenv("API_KEY", "")
MODEL    = os.getenv("MODEL", "deepseek-chat")
if not API_KEY:
    print("[警告] 未检测到 API_KEY，请复制 .env.example 为 .env 并填写")

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, temperature=0.8)
MAX_REVISIONS = 2   # 评审打回的最大重写次数

# =====================================================================
# 2. RAG 知识库：本地轻量检索（字二元组倒排索引，无需向量服务）
#    用法：在 server.py 同级建 knowledge 文件夹，放入 .txt/.md/.pdf 课件
# =====================================================================
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
CHUNK_SIZE, CHUNK_OVERLAP = 200, 40

def _read_file(path: str) -> str:
    if path.lower().endswith((".txt", ".md")):
        return open(path, encoding="utf-8", errors="ignore").read()
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        except ImportError:
            print(f"[提示] 读取 {os.path.basename(path)} 需要: pip install pypdf")
    return ""

def load_knowledge() -> List[str]:
    chunks = []
    if not os.path.isdir(KNOWLEDGE_DIR):
        return chunks
    for fn in sorted(os.listdir(KNOWLEDGE_DIR)):
        text = _read_file(os.path.join(KNOWLEDGE_DIR, fn))
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            c = text[i:i + CHUNK_SIZE].strip()
            if len(c) > 20:
                chunks.append(c)
    return chunks

KB_CHUNKS = load_knowledge()

def _bigrams(s: str) -> set:
    return set(s[i:i + 2] for i in range(len(s) - 1))

def rag_retrieve(query: str, top_k: int = 2) -> str:
    if not KB_CHUNKS:
        return ("知识库为空：请把《通信原理》笔记(.txt/.md)或课件(.pdf，需 pip install pypdf)"
                "放进 knowledge 文件夹后重启服务。")
    q = _bigrams(query.lower())
    scored = sorted(KB_CHUNKS,
                    key=lambda c: len(q & _bigrams(c.lower())), reverse=True)
    best = scored[:top_k]
    if not (q & _bigrams(best[0].lower())):
        return "知识库中未找到相关内容，换个关键词试试。"
    return "\n---\n".join(best)

# =====================================================================
# 3. 工具区：智能体可自主调用的能力（复制 @tool 函数即可扩充）
# =====================================================================
@tool
def calculator(expression: str) -> str:
    """当用户需要计算数学表达式时调用，例如 '23*47'、'(3+5)/2'"""
    import math
    env = {"sqrt": math.sqrt, "log2": math.log2, "log10": math.log10,
           "pi": math.pi, "e": math.e, "abs": abs, "round": round}
    try:
        return str(eval(expression, {"__builtins__": {}}, env))
    except Exception as e:
        return f"计算出错：{e}"

@tool
def rag_search(query: str) -> str:
    """当用户询问通信原理等专业课程内容时调用，从知识库检索相关笔记"""
    return rag_retrieve(query)

@tool
def get_time() -> str:
    """当用户询问当前日期或时间时调用"""
    return "现在是 " + datetime.now().strftime("%Y-%m-%d %H:%M")

tools = [calculator, rag_search, get_time]
llm_with_tools = llm.bind_tools(tools)

# =====================================================================
# 4. LangGraph 状态图：多智能体（生成 → 评审 → 打回重写 / 通过）
# =====================================================================
class AgentState(TypedDict):
    # 关键：Annotated + operator.add 表示"追加消息"而不是"覆盖"
    # 不加这个，ToolNode 的工具消息会覆盖全部历史，导致大模型报 400
    messages: Annotated[List[Any], operator.add]
    revision: int        # 已重写轮次（防死循环）
    critique: str        # 评审意见

def agent_node(state: AgentState):
    """生成节点：大模型根据人设+历史，决定直接回复还是调用工具"""
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp], "revision": state.get("revision", 0) + 1}

def route_after_agent(state: AgentState) -> str:
    """路由1：要调工具 → 工具节点；否则送去评审"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "reflect"

def reflector_node(state: AgentState):
    """评审节点（第二个智能体）：只挑毛病不改写，PASS 或 REVISE:建议"""
    prompt = (
        "你是智能体的评审员。检查上一条助手回复是否：1) 符合人设语气；"
        "2) 准确回答了用户问题；3) 无格式混乱、无空话。只输出一行：PASS 或 REVISE:具体修改建议"
    )
    res = llm.invoke([SystemMessage(content=prompt)] + state["messages"][-4:])
    critique = res.content.strip()
    if critique.upper().startswith("PASS"):
        return {"critique": "PASS"}
    return {"critique": critique,
            "messages": [HumanMessage(
                content=f"【评审意见】{critique}。请据此修改上一条回复，只输出修改后的最终版本。")]}

def route_after_reflect(state: AgentState) -> str:
    """路由2：评审通过或重写次数用完 → 结束；否则打回生成节点重写"""
    if state["revision"] >= MAX_REVISIONS or state["critique"].upper().startswith("PASS"):
        return END
    return "agent"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_node("reflect", reflector_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "reflect": "reflect"})
graph.add_edge("tools", "agent")
graph.add_conditional_edges("reflect", route_after_reflect, {"agent": "agent", END: END})
app_graph = graph.compile()

# =====================================================================
# 5. Web 服务
# =====================================================================
app = FastAPI(title="智能体 v5 · 多智能体 + RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatReq(BaseModel):
    message: str
    config: dict
    history: list

def build_system_prompt(c: dict) -> str:
    return (
        f"你是{c.get('name','小灵')}，{c.get('role','AI助手')}。" + "\n"
        f"【性格设定】{c.get('persona','')}" + "\n"
        f"【语气风格】{c.get('tone','亲切自然')}" + "\n"
        f"【性格参数】幽默感{c.get('humor',50)}/100，热情度{c.get('warm',50)}/100，"
        f"健谈程度{c.get('talk',50)}/100。" + "\n"
        "请始终以上述人设与用户对话，不要跳出角色。"
        "需要计算、查课程知识或看时间时，主动调用对应工具。"
        "回答中涉及公式请用 $$...$$ 包裹，便于网页渲染。"
    )

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "langgraph_chat.html"))

@app.post("/chat")
def chat(req: ChatReq):
    c = req.config
    messages: List[Any] = [SystemMessage(content=build_system_prompt(c))]
    for m in req.history[-20:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=req.message))

    try:
        final_state = app_graph.invoke({"messages": messages, "revision": 0, "critique": ""})
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print("[后端错误]", err)
        return {"reply": "（后端调用大模型出错）" + err, "tool_used": None}

    reply = ""
    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage) and m.content:
            reply = m.content
            break
    tool_used = None
    for m in final_state["messages"]:
        tc = getattr(m, "tool_calls", None)
        if tc:
            tool_used = tc[0]["name"]
    return {"reply": reply, "tool_used": tool_used}

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print(f" 知识库：已加载 {len(KB_CHUNKS)} 个文本块（knowledge 文件夹）")
    print(" 智能体已启动：http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
