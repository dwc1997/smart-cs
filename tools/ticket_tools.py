"""工单管理工具

创建、查询、更新客服工单。
"""

import json
import logging
import time
import uuid
from typing import Any, Dict

from tools.registry import registry
from core.memory_manager import memory_manager

logger = logging.getLogger(__name__)


def create_ticket_tool(
    user_id: str,
    session_id: str,
    category: str,
    subject: str,
    description: str,
    priority: str = "normal",
) -> Dict[str, Any]:
    """创建工单。"""
    ticket_id = f"TK-{uuid.uuid4().hex[:8].upper()}"

    try:
        memory_manager.db.create_ticket(
            ticket_id=ticket_id,
            user_id=user_id,
            session_id=session_id,
            category=category,
            subject=subject,
            description=description,
            priority=priority,
        )

        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"工单 {ticket_id} 已创建，我们会尽快处理。",
        }
    except Exception as e:
        logger.error(f"Create ticket failed: {e}")
        return {"success": False, "error": str(e)}


def update_ticket_tool(
    ticket_id: str,
    status: str = None,
    metadata: dict = None,
) -> Dict[str, Any]:
    """更新工单状态。"""
    try:
        memory_manager.db.update_ticket(ticket_id, status=status, metadata=metadata)
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"工单 {ticket_id} 已更新。",
        }
    except Exception as e:
        logger.error(f"Update ticket failed: {e}")
        return {"success": False, "error": str(e)}


def query_ticket_tool(
    user_id: str = None,
    ticket_id: str = None,
    status: str = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """查询工单。"""
    try:
        if ticket_id:
            # 查询特定工单
            with memory_manager.db._lock:
                cursor = memory_manager.db._conn.execute(
                    "SELECT id, user_id, category, subject, description, status, priority, "
                    "created_at, updated_at, resolved_at FROM tickets WHERE id = ?",
                    (ticket_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return {"success": True, "count": 0, "results": [], "message": "未找到该工单"}
                return {
                    "success": True,
                    "count": 1,
                    "results": [{
                        "id": row[0], "user_id": row[1], "category": row[2],
                        "subject": row[3], "description": row[4], "status": row[5],
                        "priority": row[6], "created_at": row[7], "updated_at": row[8],
                        "resolved_at": row[9],
                    }],
                }
        elif user_id:
            tickets = memory_manager.db.get_user_tickets(user_id, status=status, limit=limit)
            return {
                "success": True,
                "count": len(tickets),
                "results": tickets,
            }
        else:
            return {"success": False, "error": "需要提供 user_id 或 ticket_id"}
    except Exception as e:
        logger.error(f"Query ticket failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# 注册工具
# ============================================================

CREATE_TICKET_SCHEMA = {
    "name": "create_ticket",
    "description": "创建客服工单，记录用户问题以便后续跟进处理",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "用户 ID"},
            "session_id": {"type": "string", "description": "会话 ID"},
            "category": {
                "type": "string",
                "description": "工单类别",
                "enum": ["query", "complaint", "technical", "business", "billing", "other"],
            },
            "subject": {"type": "string", "description": "工单主题"},
            "description": {"type": "string", "description": "问题描述"},
            "priority": {
                "type": "string",
                "description": "优先级",
                "enum": ["low", "normal", "high", "urgent"],
                "default": "normal",
            },
        },
        "required": ["user_id", "session_id", "category", "subject", "description"],
    },
}

UPDATE_TICKET_SCHEMA = {
    "name": "update_ticket",
    "description": "更新工单状态（处理中、已解决、已关闭等）",
    "parameters": {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "description": "工单 ID"},
            "status": {
                "type": "string",
                "description": "新状态",
                "enum": ["open", "in_progress", "resolved", "closed"],
            },
        },
        "required": ["ticket_id"],
    },
}

QUERY_TICKET_SCHEMA = {
    "name": "query_ticket",
    "description": "查询工单信息，可按用户 ID 或工单 ID 查询",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "用户 ID（按用户查询）"},
            "ticket_id": {"type": "string", "description": "工单 ID（精确查询）"},
            "status": {
                "type": "string",
                "description": "按状态筛选",
                "enum": ["open", "in_progress", "resolved", "closed"],
            },
            "limit": {"type": "integer", "description": "返回数量", "default": 10},
        },
    },
}

registry.register(
    name="create_ticket",
    toolset="ticket",
    schema=CREATE_TICKET_SCHEMA,
    handler=lambda args, **kw: create_ticket_tool(
        user_id=args.get("user_id", kw.get("user_id", "")),
        session_id=args.get("session_id", kw.get("session_id", "")),
        category=args.get("category", "other"),
        subject=args.get("subject", ""),
        description=args.get("description", ""),
        priority=args.get("priority", "normal"),
    ),
    description="创建客服工单",
)

registry.register(
    name="update_ticket",
    toolset="ticket",
    schema=UPDATE_TICKET_SCHEMA,
    handler=lambda args, **kw: update_ticket_tool(
        ticket_id=args.get("ticket_id", ""),
        status=args.get("status"),
    ),
    description="更新工单状态",
)

registry.register(
    name="query_ticket",
    toolset="ticket",
    schema=QUERY_TICKET_SCHEMA,
    handler=lambda args, **kw: query_ticket_tool(
        user_id=args.get("user_id", kw.get("user_id")),
        ticket_id=args.get("ticket_id"),
        status=args.get("status"),
        limit=args.get("limit", 10),
    ),
    description="查询工单",
)
