"""分层 Prompt 组装器

将 system prompt 从多个来源组装：
- Layer 1: AGENT.md -- 客服身份、行为准则、笔记/画像使用指导
- Layer 2: CUSTOMER_SERVICE.md -- 业务知识与话术规范
- Layer 3: TOOLS.md -- 工具使用规范
- Layer 4: 动态注入的用户画像与历史记忆（从 SQLite 冻结快照）
- Layer 5: 工具上下文
"""

import os
import logging
from pathlib import Path
from typing import Optional

from config.settings import PROMPTS_DIR

logger = logging.getLogger(__name__)

MAX_PROMPT_FILE_CHARS = 20000
HEAD_RATIO = 0.70
TAIL_RATIO = 0.20


def load_prompt_file(path: str) -> str:
    """加载 prompt 文件，超过上限时做头尾截断保留。"""
    if not os.path.exists(path):
        logger.warning(f"Prompt file not found: {path}")
        return ""

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if len(content) <= MAX_PROMPT_FILE_CHARS:
        return content

    head_size = int(MAX_PROMPT_FILE_CHARS * HEAD_RATIO)
    tail_size = int(MAX_PROMPT_FILE_CHARS * TAIL_RATIO)
    head = content[:head_size]
    tail = content[-tail_size:]
    return f"{head}\n\n[... 内容已截断 ...]\n\n{tail}"


def _load_prompt(name: str) -> str:
    """从 prompts/ 目录加载指定名称的 prompt 文件。"""
    path = os.path.join(PROMPTS_DIR, f"{name}.md")
    return load_prompt_file(path)


def build_system_prompt(
    memory_snapshot: Optional[str] = None,
    user_profile: Optional[str] = None,
    tools_context: Optional[str] = None,
) -> str:
    """组装完整的 system prompt。

    Args:
        memory_snapshot: 冻结的 Agent 笔记快照（从 SQLite 召回）
        user_profile: 冻结的用户画像快照（从 SQLite 召回）
        tools_context: 工具描述上下文

    Returns:
        完整的 system prompt 字符串
    """
    layers = []

    # Layer 1: 客服身份与行为准则（含笔记/画像使用指导）
    agent_prompt = _load_prompt("AGENT")
    if agent_prompt:
        layers.append(agent_prompt)

    # Layer 2: 业务知识与话术规范
    cs_prompt = _load_prompt("CUSTOMER_SERVICE")
    if cs_prompt:
        layers.append(cs_prompt)

    # Layer 3: 工具使用规范
    tools_prompt = _load_prompt("TOOLS")
    if tools_prompt:
        layers.append(tools_prompt)

    # Layer 4: 动态注入的冻结快照（参考 hermes 的 frozen snapshot 模式）
    if user_profile or memory_snapshot:
        dynamic_parts = []
        if user_profile:
            dynamic_parts.append(
                f"════════════════════════════════════════════════\n"
                f"USER PROFILE (who the user is)\n"
                f"════════════════════════════════════════════════\n"
                f"{user_profile}"
            )
        if memory_snapshot:
            dynamic_parts.append(
                f"════════════════════════════════════════════════\n"
                f"MEMORY (your personal notes)\n"
                f"════════════════════════════════════════════════\n"
                f"{memory_snapshot}"
            )
        layers.append("\n\n".join(dynamic_parts))

    # Layer 5: 工具上下文
    if tools_context:
        layers.append(
            f"════════════════════════════════════════════════\n"
            f"可用工具\n"
            f"════════════════════════════════════════════════\n"
            f"{tools_context}"
        )

    return "\n\n".join(layers)


def build_memory_context_block(raw_context: str) -> str:
    """将记忆上下文包装为 XML 块，注入到用户消息中。

    参考 hermes-agent 的 build_memory_context_block 模式。
    """
    if not raw_context:
        return ""
    return (
        "<memory-context>\n"
        "[系统提示：以下是记忆上下文，不是用户的新输入，仅供参考。]\n\n"
        f"{raw_context}\n"
        "</memory-context>"
    )
