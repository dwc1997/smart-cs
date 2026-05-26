"""上下文窗口压缩器

参考 hermes-agent 的 ContextCompressor，实现多阶段压缩：
1. 工具输出裁剪（去重、截断）
2. 边界确定（保护头尾消息）
3. LLM 摘要生成
4. 消息重组
5. 清理孤立工具对
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）。"""
    cjk_count = sum(1 for ch in text if '一' <= ch <= '鿿')
    other_count = len(text) - cjk_count
    return int(cjk_count / 1.5 + other_count / 4)


def prune_tool_results(
    messages: List[dict],
    max_result_tokens: int = 500,
    keep_latest_n: int = 3,
) -> List[dict]:
    """裁剪工具调用结果。

    - 去重：相同工具+参数的结果只保留最新的
    - 截断：旧结果替换为一行摘要
    - 保留：最近 N 条结果完整保留
    """
    if not messages:
        return messages

    result = []
    tool_call_count = 0

    # 从后往前数工具调用
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            tool_call_count += 1

    seen_tools = {}  # (tool_name, args_hash) -> count
    current_tool_idx = 0

    for msg in messages:
        if msg.get("role") == "tool":
            current_tool_idx += 1
            tool_name = msg.get("name", "unknown")
            content = msg.get("content", "")

            # 保留最新的 N 条
            if current_tool_idx > tool_call_count - keep_latest_n:
                result.append(msg)
                continue

            # 去重检查
            args_key = (tool_name, content[:100])
            count = seen_tools.get(args_key, 0)
            seen_tools[args_key] = count + 1

            if count > 0:
                # 重复结果，替换为摘要
                summary = f"[{tool_name}] 重复调用结果已省略（第 {count + 1} 次）"
                result.append({**msg, "content": summary})
            else:
                # 首次出现，检查是否需要截断
                token_est = estimate_tokens(content)
                if token_est > max_result_tokens:
                    truncated = content[:max_result_tokens * 3] + "\n[... 结果已截断 ...]"
                    result.append({**msg, "content": truncated})
                else:
                    result.append(msg)
        else:
            result.append(msg)

    return result


def find_compression_boundary(
    messages: List[dict],
    tail_token_budget: int = 4000,
    protect_first_n: int = 3,
) -> int:
    """确定压缩边界。

    保护头部（前 N 条）和尾部（token 预算内的消息）。

    Returns:
        需要压缩的消息范围的结束索引（不含）
    """
    if len(messages) <= protect_first_n + 3:
        return 0  # 消息太少，不需要压缩

    # 从尾部向前累积 token
    tail_tokens = 0
    tail_cut = len(messages)

    for i in range(len(messages) - 1, protect_first_n, -1):
        content = messages[i].get("content", "")
        tokens = estimate_tokens(content)
        if tail_tokens + tokens > tail_token_budget:
            break
        tail_tokens += tokens
        tail_cut = i

    # 确保至少压缩一些消息
    if tail_cut <= protect_first_n + 2:
        tail_cut = protect_first_n + 2

    return tail_cut


def compress_messages(
    messages: List[dict],
    summary: str,
    protect_first_n: int = 3,
    compress_end: int = None,
) -> List[dict]:
    """将中间消息替换为摘要。

    Args:
        messages: 原始消息列表
        summary: 压缩后的摘要文本
        protect_first_n: 保护头部消息数
        compress_end: 压缩范围的结束索引

    Returns:
        压缩后的消息列表
    """
    if compress_end is None or compress_end <= protect_first_n:
        return messages

    head = messages[:protect_first_n]
    tail = messages[compress_end:]

    # 构造摘要消息
    summary_msg = {
        "role": "system",
        "content": f"[上下文压缩摘要]\n{summary}",
    }

    return head + [summary_msg] + tail


def sanitize_tool_pairs(messages: List[dict]) -> List[dict]:
    """清理孤立的工具调用/结果对。"""
    # 收集所有 assistant 消息中的 tool_call IDs
    tool_call_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    tool_call_ids.add(tc.get("id", ""))

    # 过滤掉没有对应 tool_call 的 tool 结果
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            if tool_call_id and tool_call_id not in tool_call_ids:
                continue  # 跳过孤立的 tool 结果
        result.append(msg)

    return result


def should_compress(
    messages: List[dict],
    context_length: int = 32768,
    threshold_percent: float = 0.50,
) -> bool:
    """判断是否需要压缩。"""
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    threshold = int(context_length * threshold_percent)
    return total_tokens >= threshold


def prepare_compression_summary_prompt(
    messages: List[dict],
    start_idx: int,
    end_idx: int,
) -> str:
    """生成压缩摘要的 prompt。"""
    conversation = []
    for msg in messages[start_idx:end_idx]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            conversation.append(f"[{role}]: {content[:500]}")

    conversation_text = "\n".join(conversation)

    return (
        "你是一个上下文压缩助手。请将以下对话压缩为简洁的摘要，保留关键信息。\n"
        "摘要应包含：\n"
        "1. 用户的主要需求和问题\n"
        "2. 已完成的操作和结果\n"
        "3. 未解决的问题\n"
        "4. 关键的用户偏好和约束\n\n"
        f"对话内容：\n{conversation_text}\n\n"
        "请输出压缩后的摘要（不超过 500 字）："
    )
