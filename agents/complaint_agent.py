"""投诉智能体

处理投诉类意图：不满、投诉、要求赔偿等。
使用 knowledge + ticket + memory 工具，会自动创建工单。
"""

import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

from agents.base_react_agent import BaseReactSubAgent

logger = logging.getLogger(__name__)


class ComplaintAgent(BaseReactSubAgent):
    name = "complaint"
    description = "投诉智能体：处理用户投诉，自动创建工单"

    allowed_toolsets = ["knowledge", "ticket", "memory"]

    # 投诉场景稍微高一点温度，让回复更有同理心
    temperature = 0.4
    max_tokens = 2048
    max_iterations = 15

    agent_system_prefix = (
        "你是一个投诉处理专员。用户正在投诉或表达不满，你需要：\n"
        "1. 先表示理解和歉意，安抚用户情绪\n"
        "2. 使用知识库查找相关政策和解决方案\n"
        "3. 主动使用工单工具创建投诉工单，确保问题被记录和跟踪\n"
        "4. 给出明确的处理方案和时间预期\n"
        "请始终保持专业、耐心和同理心。"
    )

    async def run(
        self,
        messages: List[dict],
        system_prompt: str = "",
    ) -> dict:
        """投诉智能体执行后，如果没有创建工单则自动补建。"""
        result = await super().run(messages, system_prompt)

        # 检查是否已创建工单，如果没有则自动创建
        if not self._has_ticket_created(result):
            ticket_id = self._auto_create_ticket(messages, result)
            if ticket_id:
                result.setdefault("tool_calls", []).append({
                    "tool": "create_ticket",
                    "args": {"auto_created": True, "ticket_id": ticket_id},
                    "result": f"自动创建工单: {ticket_id}",
                })
        return result

    async def run_streaming(
        self,
        messages: List[dict],
        system_prompt: str = "",
    ) -> AsyncGenerator[dict, None]:
        """流式执行，结束后检查工单。"""
        collected_result = None
        async for event in super().run_streaming(messages, system_prompt):
            if event.get("type") == "final":
                collected_result = event
            yield event

        # 流结束后检查工单
        if collected_result and not self._has_ticket_created(collected_result):
            ticket_id = self._auto_create_ticket(messages, collected_result)
            if ticket_id:
                yield {
                    "type": "tool_call",
                    "name": "create_ticket",
                    "args": {"auto_created": True},
                }
                yield {
                    "type": "tool_result",
                    "name": "create_ticket",
                    "result": f"自动创建工单: {ticket_id}",
                }

    @staticmethod
    def _has_ticket_created(result: dict) -> bool:
        """检查结果中是否包含工单创建操作。"""
        tool_calls = result.get("tool_calls", [])
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("tool") == "create_ticket":
                return True
        return False

    @staticmethod
    def _auto_create_ticket(messages: List[dict], result: dict) -> Optional[str]:
        """自动创建投诉工单。"""
        from tools.registry import registry

        # 提取用户问题摘要
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # 尝试调用 create_ticket 工具
        entry = registry.get_entry("create_ticket")
        if not entry:
            logger.warning("create_ticket tool not found, skip auto-create")
            return None

        try:
            ticket_result = entry.handler({
                "title": f"投诉: {user_query[:100]}",
                "description": user_query,
                "priority": "high",
                "category": "complaint",
                "auto_created": True,
            })
            if isinstance(ticket_result, str):
                data = json.loads(ticket_result)
            else:
                data = ticket_result
            ticket_id = data.get("ticket_id", "unknown")
            logger.info(f"Auto-created complaint ticket: {ticket_id}")
            return ticket_id
        except Exception as e:
            logger.error(f"Auto-create ticket failed: {e}")
            return None
