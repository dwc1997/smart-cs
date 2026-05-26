"""主编排 Agent

负责意图识别、子 Agent 调度、最终回答生成。
是整个智能客服系统的核心调度器。
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from agents.retrieval_agent import RetrievalAgent
from agents.calculation_agent import CalculationAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from state.cs_states import (
    CSAssistantState, IntentType, IntentAnalysisResult,
    AgentMode, WorkflowStep, ToolTask,
)
from core.prompt_builder import build_system_prompt
from config.settings import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, DEFAULT_TIMEOUT


def _get_proxy_url() -> str | None:
    """获取 HTTP/HTTPS 代理地址，忽略 SOCKS 代理。"""
    proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if proxy_url and proxy_url.startswith("socks"):
        return None
    return proxy_url

logger = logging.getLogger(__name__)

# 意图分析 prompt
INTENT_ANALYSIS_PROMPT = """你是一个意图识别助手。分析用户的输入，判断用户意图并提取关键信息。

请以 JSON 格式返回：
{
    "intent_type": "query|complaint|technical|business|billing|general|human_transfer",
    "confidence": 0.0-1.0,
    "summary": "用户需求简要描述",
    "entities": {"key": "value"},
    "requires_clarification": false
}

意图类型说明：
- query: 业务查询（话费、流量、套餐信息）
- complaint: 投诉（服务质量、费用争议）
- technical: 技术支持（网络故障、设备问题）
- business: 业务办理（开通/取消服务）
- billing: 账单相关（账单争议、费用疑问）
- general: 一般咨询（不涉及具体业务）
- human_transfer: 要求转人工客服

用户输入：{user_input}"""


class CustomerServiceAgent(BaseAgent):
    """主编排 Agent"""

    name = "customer_service"
    description = "主编排 Agent，负责意图识别和子 Agent 调度"
    tools = ["knowledge_search", "faq_lookup", "create_ticket", "update_ticket",
             "query_ticket", "calculator", "billing_query", "save_memory",
             "recall_memory", "search_history", "ask_clarification"]

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.calculation_agent = CalculationAgent()
        self.kg_agent = KnowledgeGraphAgent()
        proxy = _get_proxy_url()
        self._llm = ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL,
            temperature=0.1,
            max_tokens=500,
            timeout=30.0,
            http_client=httpx.Client(proxy=proxy, timeout=30.0),
            http_async_client=httpx.AsyncClient(proxy=proxy, timeout=30.0),
        )
        self._answer_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL,
            temperature=0.3,
            max_tokens=2048,
            timeout=60.0,
            http_client=httpx.Client(proxy=proxy, timeout=60.0),
            http_async_client=httpx.AsyncClient(proxy=proxy, timeout=60.0),
        )

    async def run(self, state: CSAssistantState) -> CSAssistantState:
        """主编排流程。"""
        current_step = state.get("current_step", WorkflowStep.START)

        if current_step == WorkflowStep.START:
            state = await self._classify_intent(state)

        if state.get("current_step") == WorkflowStep.ROUTING:
            state = await self._route_and_execute(state)

        if state.get("current_step") == WorkflowStep.ANSWERING:
            state = await self._generate_answer(state)

        return state

    async def _classify_intent(self, state: CSAssistantState) -> CSAssistantState:
        """意图分类。"""
        user_query = state.get("user_query", "")

        try:
            prompt = INTENT_ANALYSIS_PROMPT.format(user_input=user_query)
            ai_msg: AIMessage = await self._llm.ainvoke([HumanMessage(content=prompt)])
            content = ai_msg.content or ""

            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
            else:
                intent_data = {
                    "intent_type": "general",
                    "confidence": 0.5,
                    "summary": user_query[:100],
                    "entities": {},
                    "requires_clarification": False,
                }

            intent_result = IntentAnalysisResult(
                intent_type=intent_data.get("intent_type", "general"),
                confidence=intent_data.get("confidence", 0.5),
                summary=intent_data.get("summary", ""),
                entities=intent_data.get("entities", {}),
                requires_clarification=intent_data.get("requires_clarification", False),
            )

            return {
                **state,
                "intent_analysis": intent_result,
                "confirmed_intent": intent_result["intent_type"],
                "current_step": WorkflowStep.ROUTING,
            }

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                **state,
                "intent_analysis": IntentAnalysisResult(
                    intent_type="general",
                    confidence=0.3,
                    summary=user_query[:100],
                    entities={},
                    requires_clarification=False,
                ),
                "confirmed_intent": "general",
                "current_step": WorkflowStep.ROUTING,
                "error_messages": state.get("error_messages", []) + [f"意图分析失败: {e}"],
            }

    async def _route_and_execute(self, state: CSAssistantState) -> CSAssistantState:
        """根据意图路由到子 Agent 执行。"""
        intent = state.get("confirmed_intent", "general")

        # 转人工
        if intent == "human_transfer":
            return {
                **state,
                "current_mode": AgentMode.DIRECT_ANSWER,
                "current_step": WorkflowStep.ANSWERING,
            }

        # 投诉和技术问题：先检索再处理
        if intent in ("complaint", "technical"):
            state["retrieval_queries"] = [state.get("user_query", "")]
            state = await self.retrieval_agent.run(state)
            # 投诉自动创建工单
            if intent == "complaint":
                user_id = state.get("user_id", "anonymous")
                session_id = state.get("session_id", "")
                self.execute_tool("create_ticket", {
                    "user_id": user_id,
                    "session_id": session_id,
                    "category": "complaint",
                    "subject": state.get("user_query", "")[:50],
                    "description": state.get("user_query", ""),
                    "priority": "high",
                })
            return {**state, "current_step": WorkflowStep.ANSWERING}

        # 业务查询
        if intent == "query":
            state["retrieval_queries"] = [state.get("user_query", "")]
            state = await self.retrieval_agent.run(state)
            return {**state, "current_step": WorkflowStep.ANSWERING}

        # 账单相关
        if intent == "billing":
            state = await self.calculation_agent.run(state)
            return {**state, "current_step": WorkflowStep.ANSWERING}

        # 业务办理
        if intent == "business":
            state["retrieval_queries"] = [state.get("user_query", "")]
            state = await self.retrieval_agent.run(state)
            return {**state, "current_step": WorkflowStep.ANSWERING}

        # 一般咨询
        state["retrieval_queries"] = [state.get("user_query", "")]
        state = await self.retrieval_agent.run(state)
        return {**state, "current_step": WorkflowStep.ANSWERING}

    async def _generate_answer(self, state: CSAssistantState) -> CSAssistantState:
        """基于检索结果生成最终回答。"""
        user_query = state.get("user_query", "")
        retrieval_results = state.get("retrieval_results", [])
        user_profile = state.get("user_profile")
        intent = state.get("confirmed_intent", "general")

        # 构建上下文
        context_parts = []
        if retrieval_results:
            for i, r in enumerate(retrieval_results[:5], 1):
                if r.get("answer"):
                    context_parts.append(f"FAQ {i}: Q: {r.get('question', '')} A: {r['answer']}")
                elif r.get("content"):
                    context_parts.append(f"知识库 {i}: {r.get('title', '')} - {r['content'][:300]}")

        context = "\n".join(context_parts) if context_parts else "未找到相关信息"

        profile_str = ""
        if user_profile:
            profile_str = json.dumps(user_profile, ensure_ascii=False, default=str)[:500]

        try:
            system_prompt = build_system_prompt(
                memory_snapshot=state.get("memory_context"),
                user_profile=profile_str,
            )

            answer_prompt = (
                f"用户问题：{user_query}\n"
                f"意图类型：{intent}\n"
                f"检索结果：\n{context}\n\n"
                f"请用专业、友善的语气回答用户问题。"
            )

            ai_msg: AIMessage = await self._answer_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=answer_prompt),
            ])
            answer = ai_msg.content or ""

            return {
                **state,
                "messages": [AIMessage(content=answer)],
                "current_step": WorkflowStep.COMPLETED,
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            fallback = "非常抱歉，系统暂时无法处理您的问题。请稍后重试或联系人工客服。"
            return {
                **state,
                "messages": [AIMessage(content=fallback)],
                "current_step": WorkflowStep.ERROR,
                "error_messages": state.get("error_messages", []) + [str(e)],
            }
