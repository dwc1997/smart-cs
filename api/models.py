"""API 请求/响应模型"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="用户查询")
    session_id: Optional[str] = Field(None, description="会话 ID")
    user_id: Optional[str] = Field(None, description="用户 ID")


class ResumeRequest(BaseModel):
    """恢复中断请求"""
    session_id: str = Field(..., description="会话 ID")
    decision: Dict[str, Any] = Field(..., description="用户决策")


class QueryResponse(BaseModel):
    """查询响应"""
    status: str = "success"
    session_id: str = ""
    intent: Optional[str] = None
    confidence: Optional[float] = None
    answer: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    iterations: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    tools_count: int = 0
