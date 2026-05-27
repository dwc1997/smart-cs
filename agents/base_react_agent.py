"""Base ReAct Agent

封装 ReActAgent，为每个子智能体提供隔离的工具集和可独立配置的参数。
所有专用智能体继承此类。
"""

import logging
from typing import AsyncGenerator, Dict, List, Optional

from core.react_agent import ReActAgent

logger = logging.getLogger(__name__)


class BaseReactSubAgent:
    """ReAct 子智能体基类

    每个子智能体定义自己允许的 toolsets（工具集），
    运行时只暴露这些工具给 ReAct 循环。
    子类可覆盖 class-level 属性来定制行为。
    """

    name: str = "base"
    description: str = ""

    # 子类覆盖：该智能体允许使用的工具集名称
    allowed_toolsets: List[str] = ["knowledge", "memory"]

    # 子类可覆盖的 LLM 参数
    temperature: float = 0.3
    max_tokens: int = 2048
    max_iterations: int = 15

    # 可选：该智能体专属 system prompt 前缀（会拼接到用户传入的 system_prompt 前面）
    agent_system_prefix: str = ""

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ):
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if max_iterations is not None:
            self.max_iterations = max_iterations

    def get_allowed_tool_names(self) -> Optional[List[str]]:
        """返回该智能体允许使用的具体工具名列表。

        通过 registry 按 toolset 过滤，返回工具名列表。
        如果没有匹配的工具，返回 None（ReAct 会以纯对话模式运行）。
        """
        from tools.registry import registry

        tools = []
        for name, entry in registry._tools.items():
            if entry.toolset in self.allowed_toolsets:
                tools.append(name)
        return tools if tools else None

    def build_system_prompt(self, system_prompt: str = "") -> str:
        """构建最终 system prompt，拼接 agent 专属前缀。"""
        parts = []
        if self.agent_system_prefix:
            parts.append(self.agent_system_prefix)
        if system_prompt:
            parts.append(system_prompt)
        return "\n\n".join(parts) if parts else ""

    def _create_react_agent(self) -> ReActAgent:
        """创建配置好的 ReActAgent 实例。"""
        return ReActAgent(
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def run(
        self,
        messages: List[dict],
        system_prompt: str = "",
    ) -> dict:
        """同步执行（非流式）。返回与 ReActAgent.run 相同的 dict。"""
        agent = self._create_react_agent()
        tool_names = self.get_allowed_tool_names()
        final_prompt = self.build_system_prompt(system_prompt)

        logger.info(
            f"[{self.name}] Running with toolsets={self.allowed_toolsets}, "
            f"tools={tool_names}, temp={self.temperature}"
        )

        result = await agent.run(
            messages=messages,
            system_prompt=final_prompt,
            tools=tool_names,
        )
        result["agent"] = self.name
        return result

    async def run_streaming(
        self,
        messages: List[dict],
        system_prompt: str = "",
    ) -> AsyncGenerator[dict, None]:
        """流式执行。yield 与 ReActAgent.run_streaming 相同的事件类型。"""
        agent = self._create_react_agent()
        tool_names = self.get_allowed_tool_names()
        final_prompt = self.build_system_prompt(system_prompt)

        logger.info(
            f"[{self.name}] Streaming with toolsets={self.allowed_toolsets}, "
            f"tools={tool_names}, temp={self.temperature}"
        )

        async for event in agent.run_streaming(
            messages=messages,
            system_prompt=final_prompt,
            tools=tool_names,
        ):
            # 在 final 事件中注入 agent 标识
            if event.get("type") == "final":
                event["agent"] = self.name
            yield event
