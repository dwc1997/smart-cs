"""记忆工具

保存和召回长期记忆，搜索历史对话。
"""

import json
import logging
from typing import Any, Dict, List

from tools.registry import registry
from core.memory_manager import memory_manager

logger = logging.getLogger(__name__)


def save_memory_tool(
    user_id: str,
    key: str,
    value: str,
    category: str = "preference",
) -> Dict[str, Any]:
    """保存信息到长期记忆。

    Args:
        user_id: 用户 ID
        key: 记忆键（如 "preferred_contact_method"）
        value: 记忆值（如 "短信通知"）
        category: 类别（preference, info, history）
    """
    try:
        # 保存到用户画像的 preferences 中
        profile = memory_manager.get_user_profile(user_id) or {}
        preferences = profile.get("preferences", {})
        if isinstance(preferences, str):
            preferences = json.loads(preferences)

        preferences[key] = {"value": value, "category": category}

        memory_manager.save_user_profile(
            user_id=user_id,
            preferences=preferences,
        )

        return {
            "success": True,
            "message": f"已记住：{key} = {value}",
        }
    except Exception as e:
        logger.error(f"Save memory failed: {e}")
        return {"success": False, "error": str(e)}


def recall_memory_tool(
    user_id: str,
    query: str = "",
) -> Dict[str, Any]:
    """从长期记忆中召回用户相关信息。"""
    try:
        context = memory_manager.recall_user_context(user_id)
        profile = memory_manager.get_user_profile(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "profile": profile,
            "context": context,
        }
    except Exception as e:
        logger.error(f"Recall memory failed: {e}")
        return {"success": False, "error": str(e)}


def search_history_tool(
    query: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """FTS5 全文搜索历史对话。"""
    try:
        results = memory_manager.search_history(query, limit=limit)
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Search history failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# 注册工具
# ============================================================

SAVE_MEMORY_SCHEMA = {
    "name": "save_memory",
    "description": "保存用户信息到长期记忆，如偏好、个人信息等",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "用户 ID"},
            "key": {"type": "string", "description": "记忆键名"},
            "value": {"type": "string", "description": "记忆值"},
            "category": {
                "type": "string",
                "description": "类别",
                "enum": ["preference", "info", "history"],
                "default": "preference",
            },
        },
        "required": ["user_id", "key", "value"],
    },
}

RECALL_MEMORY_SCHEMA = {
    "name": "recall_memory",
    "description": "从长期记忆中召回用户画像、历史偏好、过往对话等信息",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "用户 ID"},
            "query": {"type": "string", "description": "可选的查询关键词"},
        },
        "required": ["user_id"],
    },
}

SEARCH_HISTORY_SCHEMA = {
    "name": "search_history",
    "description": "全文搜索历史对话记录，支持中英文",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "limit": {"type": "integer", "description": "返回数量", "default": 10},
        },
        "required": ["query"],
    },
}

registry.register(
    name="save_memory",
    toolset="memory",
    schema=SAVE_MEMORY_SCHEMA,
    handler=lambda args, **kw: save_memory_tool(
        user_id=args.get("user_id", kw.get("user_id", "")),
        key=args.get("key", ""),
        value=args.get("value", ""),
        category=args.get("category", "preference"),
    ),
    description="保存长期记忆",
)

registry.register(
    name="recall_memory",
    toolset="memory",
    schema=RECALL_MEMORY_SCHEMA,
    handler=lambda args, **kw: recall_memory_tool(
        user_id=args.get("user_id", kw.get("user_id", "")),
        query=args.get("query", ""),
    ),
    description="召回长期记忆",
)

registry.register(
    name="search_history",
    toolset="memory",
    schema=SEARCH_HISTORY_SCHEMA,
    handler=lambda args, **kw: search_history_tool(
        query=args.get("query", ""),
        limit=args.get("limit", 10),
    ),
    description="搜索历史对话",
)
