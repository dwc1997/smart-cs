"""LangGraph StateGraph 定义

智能客服的 LangGraph 工作流图。
"""

import logging
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from state.cs_states import (
    CSAssistantState, IntentType, IntentAnalysisResult,
    AgentMode, WorkflowStep,
)
from agents.customer_service_agent import CustomerServiceAgent

logger = logging.getLogger(__name__)

# 全局 Agent 实例
_cs_agent = CustomerServiceAgent()


async def classify_intent_node(state: CSAssistantState) -> CSAssistantState:
    """意图分类节点。"""
    logger.info(f"[Node: classify_intent] Query: {state.get('user_query', '')[:50]}")
    return await _cs_agent._classify_intent(state)


async def confirm_intent_node(state: CSAssistantState) -> CSAssistantState:
    """意图确认节点（低置信度时触发）。"""
    logger.info("[Node: confirm_intent] Low confidence, requesting confirmation")
    intent = state.get("intent_analysis", {})
    return {
        **state,
        "current_step": WorkflowStep.CONFIRMING,
        "metadata": {
            **state.get("metadata", {}),
            "confirm_payload": {
                "predicted_intent": intent.get("intent_type", "general"),
                "confidence": intent.get("confidence", 0),
                "summary": intent.get("summary", ""),
            },
        },
    }


async def route_after_intent(state: CSAssistantState) -> CSAssistantState:
    """意图路由节点。"""
    logger.info(f"[Node: route_after_intent] Intent: {state.get('confirmed_intent', 'general')}")
    return await _cs_agent._route_and_execute(
        {**state, "current_step": WorkflowStep.ROUTING}
    )


async def generate_answer_node(state: CSAssistantState) -> CSAssistantState:
    """生成回答节点。"""
    logger.info("[Node: generate_answer]")
    return await _cs_agent._generate_answer(state)


async def retrieval_node(state: CSAssistantState) -> CSAssistantState:
    """检索节点。"""
    logger.info("[Node: retrieval]")
    return await _cs_agent.retrieval_agent.run(state)


async def calculation_node(state: CSAssistantState) -> CSAssistantState:
    """计算节点。"""
    logger.info("[Node: calculation]")
    return await _cs_agent.calculation_agent.run(state)


# ============================================================
# 条件路由函数
# ============================================================

def route_after_classify(state: CSAssistantState) -> str:
    """分类后的路由：低置信度确认，高置信度直接路由。"""
    from config.settings import INTENT_CONFIRM_THRESHOLD

    intent = state.get("intent_analysis", {})
    confidence = intent.get("confidence", 0)

    if confidence < INTENT_CONFIRM_THRESHOLD:
        return "confirm_intent"
    return "route_intent"


def route_by_intent(state: CSAssistantState) -> str:
    """根据意图类型路由到不同的处理节点。"""
    intent = state.get("confirmed_intent", "general")

    if intent == "billing":
        return "calculation"
    elif intent == "human_transfer":
        return "generate_answer"
    else:
        return "generate_answer"


def route_after_confirm(state: CSAssistantState) -> str:
    """确认后的路由（始终路由到意图路由）。"""
    return "route_intent"


# ============================================================
# 图构建
# ============================================================

def create_cs_graph() -> StateGraph:
    """创建智能客服 LangGraph StateGraph。"""
    graph = StateGraph(CSAssistantState)

    # 添加节点
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("confirm_intent", confirm_intent_node)
    graph.add_node("route_intent", route_after_intent)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("calculation", calculation_node)
    graph.add_node("generate_answer", generate_answer_node)

    # 添加边
    graph.add_edge(START, "classify_intent")

    # 分类后条件路由
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "confirm_intent": "confirm_intent",
            "route_intent": "route_intent",
        },
    )

    # 确认后路由
    graph.add_conditional_edges(
        "confirm_intent",
        route_after_confirm,
        {"route_intent": "route_intent"},
    )

    # 意图路由后条件边
    graph.add_conditional_edges(
        "route_intent",
        route_by_intent,
        {
            "calculation": "calculation",
            "generate_answer": "generate_answer",
        },
    )

    # 检索和计算后到回答
    graph.add_edge("retrieval", "generate_answer")
    graph.add_edge("calculation", "generate_answer")

    # 回答后结束
    graph.add_edge("generate_answer", END)

    return graph


# 编译图
cs_graph = create_cs_graph().compile()
