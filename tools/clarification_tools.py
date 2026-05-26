"""主动问询工具

参考 hermes-agent 的 clarify 工具，当信息不足时主动向用户提问。
支持多选和开放式两种模式。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

MAX_CHOICES = 4


def ask_clarification_tool(
    question: str,
    options: Optional[List[str]] = None,
    context: str = "",
) -> Dict[str, Any]:
    """主动向用户提问以获取更多信息。

    Args:
        question: 要问用户的问题
        options: 可选的选项列表（最多 4 个），为空时为开放式提问
        context: 补充上下文信息

    Returns:
        包含问询信息的字典，由上层处理实际的用户交互
    """
    if not question or not question.strip():
        return {"success": False, "error": "问题不能为空"}

    # 验证和清洗 options（参考 hermes 的 sanitization 逻辑）
    if options is not None:
        # 如果传入的是字符串（LLM 常见错误），尝试解析
        if isinstance(options, str):
            # 尝试 JSON 解析
            try:
                parsed = json.loads(options)
                if isinstance(parsed, list):
                    options = parsed
                else:
                    options = [options]
            except (json.JSONDecodeError, TypeError):
                # 按逗号分隔
                options = [s.strip() for s in options.split(",") if s.strip()]

        if not isinstance(options, list):
            options = None
        else:
            # 清洗：转字符串、去空白、过滤空项
            cleaned = []
            for opt in options:
                s = str(opt).strip()
                if s:
                    cleaned.append(s)
            options = cleaned[:MAX_CHOICES] if cleaned else None

    result = {
        "needs_clarification": True,
        "question": question.strip(),
        "context": context,
    }

    if options:
        result["options"] = options
        result["has_other"] = True
        result["mode"] = "multiple_choice"
    else:
        result["mode"] = "open_ended"

    return result


def build_clarification_prompt(
    question: str,
    options: Optional[List[str]] = None,
) -> str:
    """构建给用户的问询文本。"""
    if options:
        lines = [question, ""]
        for i, opt in enumerate(options, 1):
            lines.append(f"  {i}. {opt}")
        lines.append(f"  {len(options) + 1}. 其他（请手动输入）")
        lines.append("")
        lines.append("请输入选项编号或您的回答：")
        return "\n".join(lines)
    else:
        return f"{question}\n请输入您的回答："


# ============================================================
# 注册工具
# ============================================================

ASK_CLARIFICATION_SCHEMA = {
    "name": "ask_clarification",
    "description": (
        "当用户描述模糊或信息不足时，主动向用户提问以获取更准确的信息。"
        "适用于：需求不明确、需要确认操作意图、存在多个方案需要用户选择。\n\n"
        "使用要求：\n"
        "- question：根据对话上下文生成自然的提问，不要重复用户已提供的信息\n"
        "- options：根据当前对话上下文生成 2-4 个具体的相关选项，每个选项是一个独立的字符串，用数组格式传入\n"
        "- 选项应该是用户最可能想做的事情，而不是泛泛的分类\n\n"
        "示例：用户说'帮我查一下'，你应生成 options=['查询话费余额', '查询剩余流量', '查询当前套餐']，"
        "而不是 options=['查询', '办理', '其他']"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "根据对话上下文生成的自然提问",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "根据上下文生成的 2-4 个具体选项，每个是独立的完整短语",
                "maxItems": 4,
            },
            "context": {
                "type": "string",
                "description": "补充上下文信息，帮助用户理解问题背景",
            },
        },
        "required": ["question"],
    },
}

registry.register(
    name="ask_clarification",
    toolset="interaction",
    schema=ASK_CLARIFICATION_SCHEMA,
    handler=lambda args, **kw: ask_clarification_tool(
        question=args.get("question", ""),
        options=args.get("options"),
        context=args.get("context", ""),
    ),
    description="主动向用户提问确认",
)
