"""FastAPI 应用

智能客服 API 服务，支持 SSE 流式输出。
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

from api.models import QueryRequest, ResumeRequest, QueryResponse, HealthResponse
from core.react_agent import ReActAgent
from core.memory_manager import memory_manager
from core.prompt_builder import build_system_prompt
from tools.registry import registry

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
    """SSE 流式查询端点。"""
    session_id = request.session_id or f"session-{uuid.uuid4().hex[:8]}"
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

    agent = ReActAgent()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield _to_sse_data({"type": "start", "session_id": session_id})

            async for event in agent.run_streaming(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=system_prompt,
            ):
                event_type = event.get("type", "")

                if event_type == "token":
                    yield _to_sse_data({"type": "text", "content": event["content"]})
                elif event_type == "tool_call":
                    yield _to_sse_data({
                        "type": "tool_call",
                        "name": event["name"],
                        "args": event["args"],
                    })
                elif event_type == "tool_result":
                    yield _to_sse_data({
                        "type": "tool_result",
                        "name": event["name"],
                        "result": event["result"],
                    })
                elif event_type == "clarification":
                    yield _to_sse_data({
                        "type": "interrupt",
                        "payload": event["data"],
                    })
                elif event_type == "final":
                    # 保存回答到会话
                    memory_manager.add_message(session_id, "assistant", event["content"])
                    yield _to_sse_data({
                        "type": "final",
                        "content": event["content"],
                        "session_id": session_id,
                        "iterations": event.get("iterations", 0),
                        "tool_calls": event.get("tool_calls", []),
                    })
                elif event_type == "error":
                    yield _to_sse_data({"type": "error", "error": event["error"]})

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
    """恢复中断的对话。"""
    # TODO: 实现 interrupt/resume 机制
    raise HTTPException(status_code=501, detail="Resume not yet implemented")


@app.get("/api/v1/tools")
async def list_tools():
    """列出所有可用工具。"""
    return {"tools": registry.list_tools()}
