"""统一执行引擎

将意图分类 + 路由 + ReAct 执行统一为单一入口。
API 和 CLI 都通过这个引擎执行，不再直接调用 ReActAgent。

执行流程：
  用户输入 → 意图分类 → 工具集选择 → ReAct 循环 → 输出
                ↓              ↓
          (LLM 分类)    (根据意图过滤工具)
"""

import json
import logging
import os
import re
from typing import AsyncGenerator, Dict, List, Optional, Any

import httpx
from openai import AsyncOpenAI

from config.settings import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    MAX_ITERATIONS, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT, INTENT_CONFIRM_THRESHOLD,
)
from core.react_agent import ReActAgent
from core.context_compressor import (
    should_compress, find_compression_boundary,
    prepare_compression_summary_prompt, compress_messages,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

# 意图 → 工具集映射
INTENT_TOOLSET_MAP = {
    "query": ["knowledge", "memory"],
    "complaint": ["knowledge", "ticket", "memory"],
    "technical": ["knowledge", "memory"],
    "business": ["knowledge", "ticket", "memory"],
    "billing": ["knowledge", "calculation", "memory"],
    "general": ["knowledge", "memory"],
    "human_transfer": ["memory"],
}

# 意图分类 prompt
INTENT_CLASSIFY_PROMPT = """你是一个意图分类器。根据用户输入，判断意图类型。

可选意图：
- query: 查询类（查套餐、查账单、查流量等）
- complaint: 投诉类（不满、投诉、要求赔偿等）
- technical: 技术类（网络故障、信号问题、设备问题等）
- business: 业务办理类（开通、取消、变更套餐等）
- billing: 费用类（话费、扣费、充值、退款等）
- general: 闲聊/通用类
- human_transfer: 要求转人工

只返回 JSON，不要其他内容：
{"intent": "意图类型", "confidence": 0.0-1.0, "summary": "一句话摘要"}"""


def _get_proxy_url() -> Optional[str]:
    """获取 HTTP/HTTPS 代理地址。"""
    proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if proxy_url and proxy_url.startswith("socks"):
        return None
    return proxy_url


class CSEngine:
    """统一执行引擎

    将意图分类、路由、ReAct 执行统一为单一入口。
    支持同步和流式两种模式。
    """

    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        proxy = _get_proxy_url()
        self._classify_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=30,
            http_client=httpx.AsyncClient(proxy=proxy, timeout=30),
        )
        self._model = OPENAI_MODEL

    async def classify_intent(self, user_query: str) -> Dict[str, Any]:
        """意图分类（轻量 LLM 调用）。

        Returns:
            {"intent": str, "confidence": float, "summary": str}
        """
        try:
            resp = await self._classify_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFY_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            content = resp.choices[0].message.content or ""

            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "intent": result.get("intent", "general"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "summary": result.get("summary", ""),
                }
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")

        return {"intent": "general", "confidence": 0.5, "summary": ""}

    def select_tools_for_intent(self, intent: str) -> Optional[List[str]]:
        """根据意图选择可用工具名列表。"""
        toolsets = INTENT_TOOLSET_MAP.get(intent, ["knowledge", "memory"])
        tools = []
        for name, entry in registry._tools.items():
            if entry.toolset in toolsets:
                tools.append(name)
        return tools if tools else None

    async def _maybe_compress(
        self, messages: List[dict], system_prompt: str
    ) -> List[dict]:
        """检查并执行上下文压缩。"""
        if not should_compress(messages):
            return messages

        boundary = find_compression_boundary(messages)
        if boundary <= 0:
            return messages

        # 用 LLM 生成摘要
        prompt = prepare_compression_summary_prompt(messages, 0, boundary)
        try:
            resp = await self._classify_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            summary = resp.choices[0].message.content or "（摘要生成失败）"
        except Exception as e:
            logger.warning(f"Compression summary failed: {e}")
            summary = "（摘要生成失败，已压缩中间消息）"

        return compress_messages(messages, summary, compress_end=boundary)

    async def run(
        self,
        messages: List[dict],
        system_prompt: str = "",
        user_id: str = "anonymous",
    ) -> dict:
        """统一执行入口（同步模式）。"""
        # 1. 意图分类
        user_query = self._extract_user_query(messages)
        intent_result = await self.classify_intent(user_query)
        intent = intent_result["intent"]
        logger.info(f"Intent: {intent} (confidence: {intent_result['confidence']:.2f})")

        # 2. 选择工具
        tool_filter = self.select_tools_for_intent(intent)

        # 3. 上下文压缩
        messages = await self._maybe_compress(messages, system_prompt)

        # 4. ReAct 执行
        agent = ReActAgent(
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        result = await agent.run(
            messages=messages,
            system_prompt=system_prompt,
            tools=tool_filter,
        )

        result["intent"] = intent_result
        return result

    async def run_streaming(
        self,
        messages: List[dict],
        system_prompt: str = "",
        user_id: str = "anonymous",
    ) -> AsyncGenerator[dict, None]:
        """统一执行入口（流式模式）。"""
        # 1. 意图分类
        user_query = self._extract_user_query(messages)
        intent_result = await self.classify_intent(user_query)
        intent = intent_result["intent"]
        logger.info(f"Intent: {intent} (confidence: {intent_result['confidence']:.2f})")

        yield {
            "type": "intent_classified",
            "intent": intent,
            "confidence": intent_result["confidence"],
            "summary": intent_result["summary"],
        }

        # 2. 选择工具
        tool_filter = self.select_tools_for_intent(intent)

        # 3. 上下文压缩
        messages = await self._maybe_compress(messages, system_prompt)

        # 4. ReAct 流式执行
        agent = ReActAgent(
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        async for event in agent.run_streaming(
            messages=messages,
            system_prompt=system_prompt,
            tools=tool_filter,
        ):
            yield event

    @staticmethod
    def _extract_user_query(messages: List[dict]) -> str:
        """从消息列表中提取最后一条用户消息。"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


# 全局引擎实例
cs_engine = CSEngine()
