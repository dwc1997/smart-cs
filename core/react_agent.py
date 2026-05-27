"""ReAct Agent Loop

参考 hermes-agent 的 run_conversation 循环，实现思考-行动-观察循环。
支持：
- 最大迭代数中止
- Token 预算中止
- 用户中断
- 主动问询
- 重复检测
"""

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from config.settings import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    MAX_ITERATIONS, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT,
)
from core.error_classifier import ErrorClassifier, RetryState
from core.tool_guardrails import ToolGuardrails
from tools.registry import registry


def _get_proxy_url() -> str | None:
    """获取 HTTP/HTTPS 代理地址，忽略 SOCKS 代理。"""
    proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if proxy_url and proxy_url.startswith("socks"):
        return None
    return proxy_url

logger = logging.getLogger(__name__)


def _build_openai_tools(tool_defs: List[dict]) -> List[dict]:
    """将 registry 的 JSON Schema 转换为 OpenAI function calling 格式。"""
    tools = []
    for defn in tool_defs:
        tools.append({
            "type": "function",
            "function": {
                "name": defn.get("name", ""),
                "description": defn.get("description", ""),
                "parameters": defn.get("parameters", {}),
            },
        })
    return tools


class ReActAgent:
    """ReAct 循环 Agent

    思考(LLM) → 行动(工具调用) → 观察(工具结果) → 迭代
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
        self._interrupt_requested = False
        self._iteration_count = 0

        proxy = _get_proxy_url()
        self._client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=timeout,
            http_client=httpx.AsyncClient(proxy=proxy, timeout=timeout),
        )
        self._model = OPENAI_MODEL
        self._error_classifier = ErrorClassifier()

    def request_interrupt(self):
        """请求中断循环。"""
        self._interrupt_requested = True

    async def _call_llm(self, messages: list, tools: list) -> dict:
        """调用 LLM，返回原始 API 响应 dict。

        Wraps the raw API call with ErrorClassifier-based retry logic.
        """
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        retry_state = RetryState()
        last_exc: Optional[Exception] = None

        while True:
            try:
                resp = await self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.model_dump(exclude_none=True)
            except Exception as e:
                last_exc = e
                classification = self._error_classifier.classify(e)
                attempt = retry_state.attempt(classification.category)

                if classification.should_abort:
                    logger.error(
                        f"LLM call failed (abort): [{classification.category.value}] {e}"
                    )
                    raise

                if classification.should_retry and attempt <= classification.max_retries:
                    delay = self._error_classifier.backoff_delay(classification, retry_state)
                    logger.warning(
                        f"LLM call failed [{classification.category.value}], "
                        f"retry {attempt}/{classification.max_retries} in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue

                if classification.should_compress:
                    logger.warning(
                        f"LLM call failed [{classification.category.value}]: {e}"
                    )
                    raise

                # fallback_model or other — abort
                logger.error(f"LLM call failed (unhandled): {e}")
                raise

    async def run(
        self,
        messages: List[dict],
        system_prompt: str = "",
        tools: Optional[List[str]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
    ) -> dict:
        """执行 ReAct 循环。"""
        self._interrupt_requested = False
        self._iteration_count = 0

        tool_defs = registry.get_definitions()
        if tools:
            tool_defs = [t for t in tool_defs if t.get("name") in tools]
        openai_tools = _build_openai_tools(tool_defs)

        # 保持原始 dict 格式，保留 reasoning_content
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        tool_call_history = []
        guardrails = ToolGuardrails()

        while self._iteration_count < self.max_iterations:
            if self._interrupt_requested:
                return {
                    "content": "对话已被中断。",
                    "tool_calls": tool_call_history,
                    "iterations": self._iteration_count,
                    "interrupted": True,
                    "error": None,
                }

            self._iteration_count += 1

            try:
                msg = await self._call_llm(full_messages, openai_tools)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {
                    "content": "抱歉，系统暂时无法处理您的请求，请稍后重试。",
                    "tool_calls": tool_call_history,
                    "iterations": self._iteration_count,
                    "interrupted": False,
                    "error": str(e),
                }

            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            if on_token and content:
                on_token(content)

            if not tool_calls:
                return {
                    "content": content,
                    "tool_calls": tool_call_history,
                    "iterations": self._iteration_count,
                    "interrupted": False,
                    "error": None,
                }

            # 保留完整 assistant 消息（含 reasoning_content）
            full_messages.append(msg)

            for tc in tool_calls:
                if self._interrupt_requested:
                    break

                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tc_id = tc.get("id", "")

                try:
                    tool_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                if on_tool_call:
                    on_tool_call(tool_name, tool_args)
                tool_result = registry.dispatch(tool_name, tool_args)
                if on_tool_result:
                    on_tool_result(tool_name, tool_result)
                tool_call_history.append({
                    "tool": tool_name, "args": tool_args, "result": tool_result[:1000],
                })

                # Check guardrails after executing the tool
                verdict = guardrails.check(tool_name, tool_args, tool_result)
                if verdict.warning:
                    logger.warning(f"[guardrails] {verdict.reason}")
                if not verdict.ok:
                    logger.warning(f"[guardrails] Hard stop: {verdict.reason}")
                    tool_result = json.dumps({
                        "error": f"检测到循环调用，已自动停止。原因：{verdict.reason}。请尝试不同的方法或直接回答用户。",
                    }, ensure_ascii=False)
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    })
                    break

                try:
                    result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                    if isinstance(result_data, dict) and result_data.get("needs_clarification"):
                        return {
                            "content": content or "",
                            "tool_calls": tool_call_history,
                            "iterations": self._iteration_count,
                            "interrupted": False,
                            "needs_clarification": True,
                            "clarification": result_data,
                            "error": None,
                        }
                except (json.JSONDecodeError, TypeError):
                    pass

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

        return {
            "content": "抱歉，处理过程超时。请简化您的问题或联系人工客服。",
            "tool_calls": tool_call_history,
            "iterations": self._iteration_count,
            "interrupted": False,
            "error": "max_iterations_exceeded",
        }

    async def run_streaming(
        self,
        messages: List[dict],
        system_prompt: str = "",
        tools: Optional[List[str]] = None,
    ):
        """流式执行 ReAct 循环，yield 事件。

        Yields:
            {"type": "token", "content": "..."}
            {"type": "tool_call", "name": "...", "args": {...}}
            {"type": "tool_result", "name": "...", "result": "..."}
            {"type": "clarification", "data": {...}}
            {"type": "guardrail_warning", "reason": "..."}
            {"type": "final", "content": "...", "tool_calls": [...]}
            {"type": "error", "error": "..."}
        """
        self._interrupt_requested = False
        self._iteration_count = 0

        tool_defs = registry.get_definitions()
        if tools:
            tool_defs = [t for t in tool_defs if t.get("name") in tools]
        openai_tools = _build_openai_tools(tool_defs)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        tool_call_history = []
        guardrails = ToolGuardrails()

        while self._iteration_count < self.max_iterations:
            if self._interrupt_requested:
                yield {"type": "final", "content": "对话已中断。", "interrupted": True}
                return

            self._iteration_count += 1

            try:
                msg = await self._call_llm(full_messages, openai_tools)
            except Exception as e:
                yield {"type": "error", "error": str(e)}
                return

            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            if content:
                yield {"type": "token", "content": content}

            if not tool_calls:
                yield {
                    "type": "final",
                    "content": content,
                    "tool_calls": tool_call_history,
                    "iterations": self._iteration_count,
                }
                return

            # 保留完整 assistant 消息（含 reasoning_content）
            full_messages.append(msg)

            for tc in tool_calls:
                if self._interrupt_requested:
                    yield {"type": "final", "content": "对话已中断。", "interrupted": True}
                    return

                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tc_id = tc.get("id", "")

                try:
                    tool_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                yield {"type": "tool_call", "name": tool_name, "args": tool_args}

                tool_result = registry.dispatch(tool_name, tool_args)

                yield {"type": "tool_result", "name": tool_name, "result": tool_result[:500]}

                tool_call_history.append({
                    "tool": tool_name, "args": tool_args, "result": tool_result[:1000],
                })

                # Check guardrails after executing the tool
                verdict = guardrails.check(tool_name, tool_args, tool_result)
                if verdict.warning:
                    logger.warning(f"[guardrails] {verdict.reason}")
                    yield {"type": "guardrail_warning", "reason": verdict.reason}
                if not verdict.ok:
                    logger.warning(f"[guardrails] Hard stop: {verdict.reason}")
                    tool_result = json.dumps({
                        "error": f"检测到循环调用，已自动停止。原因：{verdict.reason}。请尝试不同的方法或直接回答用户。",
                    }, ensure_ascii=False)
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    })
                    break

                try:
                    result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                    if isinstance(result_data, dict) and result_data.get("needs_clarification"):
                        yield {"type": "clarification", "data": result_data}
                        return
                except (json.JSONDecodeError, TypeError):
                    pass

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

        yield {
            "type": "final",
            "content": "处理超时，请简化问题或联系人工客服。",
            "error": "max_iterations_exceeded",
        }
