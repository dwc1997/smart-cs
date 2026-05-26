"""Agent 基类

定义所有智能体的统一接口。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from state.cs_states import CSAssistantState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent 基类"""

    name: str = "base"
    description: str = ""
    tools: List[str] = []  # 该 Agent 可用的工具集

    @abstractmethod
    async def run(self, state: CSAssistantState) -> CSAssistantState:
        """执行 Agent 逻辑，返回更新后的状态。"""
        ...

    def get_tool_definitions(self) -> List[dict]:
        """获取该 Agent 可用工具的 schema 定义。"""
        from tools.registry import registry
        return registry.get_definitions(toolsets=None)

    def execute_tool(self, name: str, args: dict, **kwargs) -> str:
        """执行工具调用。"""
        from tools.registry import registry
        if name not in self.tools:
            return f'{{"error": "Agent {self.name} 无权使用工具 {name}"}}'
        return registry.dispatch(name, args, **kwargs)
