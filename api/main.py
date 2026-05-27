"""FastAPI 应用

智能客服 API 服务，支持 SSE 流式输出。
统一通过 CSEngine 执行（意图分类 → 路由 → ReAct）。
支持 LangGraph interrupt/resume 实现人机交互确认。
"""

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command

from api.models import QueryRequest, ResumeRequest, QueryResponse, HealthResponse
from core.cs_engine import cs_engine
from core.memory_manager import memory_manager
from core.prompt_builder import build_system_prompt
from tools.registry import registry
from graphs.cs_graph import cs_graph

logger = logging.getLogger(__name__)


def _configure_langsmith() -> None:
    """配置 LangSmith 追踪（可选）。"""
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        logger.info("LangSmith tracing disabled (LANGCHAIN_API_KEY not set)")
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "smart-cs"))
    logger.info(f"LangSmith tracing enabled, project: {os.getenv('LANGCHAIN_PROJECT', 'smart-cs')}")


# ============================================================
# 初始化知识库
# ============================================================
def _init_knowledge_base():
    """初始化知识库数据库。"""
    from tools.knowledge_tools import _init_knowledge_db
    _init_knowledge_db()
    logger.info("Knowledge base initialized")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="Smart-CS 智能客服 API",
    description="基于 LangGraph 的智能客服系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    _configure_langsmith()
    _init_knowledge_base()
    logger.info("Smart-CS API started")


@app.on_event("shutdown")
async def shutdown():
    memory_manager.close()
    logger.info("Smart-CS API stopped")


def _to_sse_data(payload: Dict[str, Any]) -> str:
    """转换为 SSE 格式。"""
    data_str = json.dumps(payload, ensure_ascii=False, default=str)
    return f"data: {data_str}\n\n"


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查。"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        tools_count=len(registry.get_available_tools()),
    )


@app.post("/api/v1/query")
async def query(request: QueryRequest):
    """SSE 流式查询端点。

    使用 LangGraph cs_graph 执行，支持 interrupt/resume。
    当图在 confirm_intent 节点中断时，返回 interrupt 事件和 thread_id，
    客户端可通过 /api/v1/resume 恢复执行。
    """
    session_id = request.session_id or f"session-{uuid.uuid4().hex[:8]}"
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "anonymous"

    # 创建会话
    memory_manager.create_session(session_id, user_id=user_id)
    memory_manager.add_message(session_id, "user", request.query)

    # 冻结记忆快照
    memory_manager.freeze_snapshot(user_id)

    # 获取用户上下文
    user_context = memory_manager.recall_user_context(user_id)

    # 构建 system prompt
    system_prompt = build_system_prompt(
        memory_snapshot=memory_manager.get_frozen_memory(),
        user_profile=memory_manager.get_frozen_profile(),
    )

    # 注入记忆上下文到用户消息
    user_message = request.query
    if user_context:
        user_message = f"<memory-context>\n{user_context}\n</memory-context>\n\n{request.query}"

    # 构建图初始状态
    initial_state = {
        "user_query": user_message,
        "original_input": request.query,
        "session_id": session_id,
        "user_id": user_id,
        "current_step": "start",
        "current_mode": "direct_answer",
        "intent_confirmed": False,
        "error_messages": [],
        "planned_tasks": [],
        "tool_execution_results": [],
        "retrieval_queries": [],
        "retrieval_results": [],
        "short_term_memory": [],
        "metadata": {"system_prompt": system_prompt},
    }

    graph_config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield _to_sse_data({"type": "start", "session_id": session_id, "thread_id": thread_id})

            # 使用 cs_graph.stream 执行图
            async for chunk in cs_graph.astream(initial_state, config=graph_config, stream_mode="updates"):
                # chunk 是 {node_name: state_update} 的字典
                for node_name, node_output in chunk.items():
                    if node_name == "__interrupt__":
                        # 图被 interrupt() 暂停
                        interrupt_data = node_output
                        if isinstance(interrupt_data, (list, tuple)) and len(interrupt_data) > 0:
                            interrupt_value = interrupt_data[0].value if hasattr(interrupt_data[0], 'value') else interrupt_data[0]
                        elif hasattr(interrupt_data, 'value'):
                            interrupt_value = interrupt_data.value
                        else:
                            interrupt_value = interrupt_data

                        yield _to_sse_data({
                            "type": "interrupt",
                            "thread_id": thread_id,
                            "session_id": session_id,
                            "payload": interrupt_value,
                        })
                        logger.info(f"Graph interrupted, thread_id={thread_id}")
                        continue

                    # 普通节点输出
                    if node_name == "classify_intent" and isinstance(node_output, dict):
                        intent_analysis = node_output.get("intent_analysis", {})
                        if intent_analysis:
                            yield _to_sse_data({
                                "type": "intent",
                                "intent": intent_analysis.get("intent_type", "general"),
                                "confidence": intent_analysis.get("confidence", 0),
                                "summary": intent_analysis.get("summary", ""),
                            })

                    elif node_name == "generate_answer" and isinstance(node_output, dict):
                        answer = node_output.get("metadata", {}).get("final_answer", "")
                        if answer:
                            memory_manager.add_message(session_id, "assistant", answer)
                            yield _to_sse_data({
                                "type": "final",
                                "content": answer,
                                "session_id": session_id,
                                "thread_id": thread_id,
                            })

            yield _to_sse_data({"type": "done"})

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield _to_sse_data({"type": "error", "error": str(e)})
            yield _to_sse_data({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/resume")
async def resume(request: ResumeRequest):
    """恢复中断的对话。

    使用 LangGraph 的 Command(resume=...) 机制恢复被 interrupt() 暂停的图执行。
    客户端需要提供 thread_id（来自 interrupt 事件）和用户回复。
    """
    thread_id = request.thread_id
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    session_id = request.session_id or f"session-{uuid.uuid4().hex[:8]}"

    # 构建用户回复数据
    user_response = {
        "message": request.message,
        "confirmed_intent": request.confirmed_intent,
    }
    # 兼容旧的 decision 字段
    if request.decision:
        user_response.update(request.decision)

    graph_config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield _to_sse_data({"type": "resume_start", "thread_id": thread_id, "session_id": session_id})

            # 使用 Command(resume=...) 恢复图执行
            async for chunk in cs_graph.astream(
                Command(resume=user_response),
                config=graph_config,
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if node_name == "__interrupt__":
                        # 再次中断（理论上不应该，但做防御处理）
                        interrupt_data = node_output
                        if isinstance(interrupt_data, (list, tuple)) and len(interrupt_data) > 0:
                            interrupt_value = interrupt_data[0].value if hasattr(interrupt_data[0], 'value') else interrupt_data[0]
                        elif hasattr(interrupt_data, 'value'):
                            interrupt_value = interrupt_data.value
                        else:
                            interrupt_value = interrupt_data

                        yield _to_sse_data({
                            "type": "interrupt",
                            "thread_id": thread_id,
                            "session_id": session_id,
                            "payload": interrupt_value,
                        })
                        continue

                    # 普通节点输出
                    if node_name == "generate_answer" and isinstance(node_output, dict):
                        answer = node_output.get("metadata", {}).get("final_answer", "")
                        if answer:
                            memory_manager.add_message(session_id, "assistant", answer)
                            yield _to_sse_data({
                                "type": "final",
                                "content": answer,
                                "session_id": session_id,
                                "thread_id": thread_id,
                            })

            yield _to_sse_data({"type": "done"})

        except Exception as e:
            logger.error(f"Resume error: {e}", exc_info=True)
            yield _to_sse_data({"type": "error", "error": str(e)})
            yield _to_sse_data({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/tools")
async def list_tools():
    """列出所有可用工具。"""
    return {"tools": registry.list_tools()}
