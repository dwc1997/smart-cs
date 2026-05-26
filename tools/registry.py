"""工具注册表

参考 hermes-agent 的 ToolRegistry 模式，实现轻量版工具注册与调度。
支持工具分组（toolset）、可用性检查、自动 schema 发现。
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHECK_FN_TTL_SECONDS = 30.0


@dataclass
class ToolEntry:
    """工具注册条目"""
    name: str
    toolset: str               # 分组：knowledge, ticket, calculation, memory, interaction
    schema: dict               # OpenAI function calling 格式的 JSON Schema
    handler: Callable          # 处函数，接收 (args: dict, **kwargs) -> str
    description: str
    check_fn: Optional[Callable[[], bool]] = None  # 可用性检查
    is_async: bool = False
    max_result_size_chars: int = 50000


class ToolRegistry:
    """工具注册表单例"""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._generation: int = 0
        self._lock = threading.RLock()
        self._check_fn_cache: Dict[Callable, tuple] = {}
        self._check_fn_cache_lock = threading.Lock()

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        description: str = "",
        check_fn: Optional[Callable[[], bool]] = None,
        is_async: bool = False,
        max_result_size_chars: int = 50000,
    ) -> None:
        """注册一个工具。"""
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                logger.error(
                    f"Tool registration REJECTED: '{name}' would shadow "
                    f"existing tool from toolset '{existing.toolset}'"
                )
                return

            self._tools[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                description=description,
                check_fn=check_fn,
                is_async=is_async,
                max_result_size_chars=max_result_size_chars,
            )
            self._generation += 1
            logger.debug(f"Registered tool: {name} (toolset={toolset})")

    def deregister(self, name: str) -> None:
        """注销一个工具。"""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                self._generation += 1

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """获取工具条目。"""
        return self._tools.get(name)

    def _check_fn_cached(self, fn: Callable) -> bool:
        """带 TTL 缓存的可用性检查。"""
        now = time.monotonic()
        with self._check_fn_cache_lock:
            cached = self._check_fn_cache.get(fn)
            if cached is not None:
                ts, value = cached
                if now - ts < _CHECK_FN_TTL_SECONDS:
                    return value

        try:
            value = bool(fn())
        except Exception:
            value = False

        with self._check_fn_cache_lock:
            self._check_fn_cache[fn] = (now, value)
        return value

    def get_definitions(
        self, toolsets: Optional[List[str]] = None
    ) -> List[dict]:
        """获取工具定义列表（OpenAI function calling 格式）。"""
        definitions = []
        with self._lock:
            for entry in self._tools.values():
                if toolsets and entry.toolset not in toolsets:
                    continue
                if entry.check_fn and not self._check_fn_cached(entry.check_fn):
                    continue
                definitions.append(entry.schema)
        return definitions

    def get_available_tools(self) -> List[str]:
        """获取所有可用工具名称列表。"""
        tools = []
        with self._lock:
            for entry in self._tools.values():
                if entry.check_fn and not self._check_fn_cached(entry.check_fn):
                    continue
                tools.append(entry.name)
        return tools

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """执行工具调用。"""
        entry = self.get_entry(name)
        if not entry:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

        try:
            result = entry.handler(args, **kwargs)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, default=str)
            return str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}", exc_info=True)
            return json.dumps(
                {"error": f"工具执行失败: {type(e).__name__}: {e}"},
                ensure_ascii=False,
            )

    @property
    def generation(self) -> int:
        return self._generation

    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有已注册工具的基本信息。"""
        tools = []
        with self._lock:
            for entry in self._tools.values():
                tools.append({
                    "name": entry.name,
                    "toolset": entry.toolset,
                    "description": entry.description,
                })
        return tools


# 全局单例
registry = ToolRegistry()
