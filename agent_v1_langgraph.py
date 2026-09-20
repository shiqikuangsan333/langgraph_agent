# -*- coding: utf-8 -*-
"""
agent_v1_langgraph.py —— 第一代（单智能体 + 工具调用）
============================
结构：START -> agent(大模型+工具) -> 需要工具？-> tools -> agent -> END

这是第一代：单智能体 + 工具调用，供学习和对比第二代。运行：python main.py --v1
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

# 配置从 .env 读取（main.py 已加载）
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
API_KEY  = os.getenv("API_KEY", "")
MODEL    = os.getenv("MODEL", "deepseek-chat")
if not API_KEY:
    print("[警告] 未检测到 API_KEY，请在 .env 文件中填写")

llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, temperature=0.8)

# ---- 工具区 ----
@tool
def calculator(expression: str) -> str:
    """当用户需要计算数学表达式时调用"""
    import math
    env = {"sqrt": math.sqrt, "log2": math.log2, "log10": math.log10,
           "pi": math.pi, "e": math.e, "abs": abs, "round": round}
    try:
        return str(eval(expression, {"__builtins__": {}}, env))
    except Exception as e:
        return f"计算出错：{e}"

@tool
def query_comm_knowledge(topic: str) -> str:
    """当用户询问通信工程基础知识时调用（v1 为硬编码小知识库）"""
    kb = {
        "香农公式": "C = B*log2(1+S/N)，容量由带宽和信噪比决定。",
        "5g": "5G 三大场景：eMBB、uRLLC、mMTC。",
    }
    t = topic.lower()
    for k, v in kb.items():
        if k in t:
            return v
    return "知识库中暂无该条目。"

@tool
def get_time() -> str:
    """当用户询问当前时间时调用"""
    return "现在是 " + datetime.now().strftime("%Y-%m-%d %H:%M")

tools = [calculator, query_comm_knowledge, get_time]
llm_with_tools = llm.bind_tools(tools)

# ---- 状态图 ----
class AgentState(TypedDict):
    # 关键：Annotated + operator.add 表示"追加"而不是"覆盖"
    messages: Annotated[List[Any], operator.add]

def agent_node(state: AgentState):
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app_graph = graph.compile()

# ---- Web 服务 ----
app = FastAPI(title="智能体 v1 基础版")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatReq(BaseModel):
    message: str
    config: dict
    history: list

def build_system_prompt(c: dict) -> str:
    return (f"你是{c.get('name','小灵')}，{c.get('role','AI助手')}。"
            f"性格：{c.get('persona','')} 语气：{c.get('tone','亲切自然')}。"
            "需要计算、查知识或看时间时主动调用工具。")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "langgraph_chat.html"))

@app.post("/chat")
def chat(req: ChatReq):
    messages: List[Any] = [SystemMessage(content=build_system_prompt(req.config))]
    for m in req.history[-20:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=req.message))
    try:
        final_state = app_graph.invoke({"messages": messages})
    except Exception as e:
        return {"reply": f"（后端出错）{type(e).__name__}: {e}", "tool_used": None}
    last = final_state["messages"][-1]
    tool_used = None
    for m in final_state["messages"]:
        tc = getattr(m, "tool_calls", None)
        if tc:
            tool_used = tc[0]["name"]
    return {"reply": last.content, "tool_used": tool_used}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
