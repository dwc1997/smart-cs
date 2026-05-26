"""检索 Agent

负责知识库查询和 FAQ 匹配。
"""

import json
import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from state.cs_states import CSAssistantState, AgentMode, WorkflowStep

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """知识库检索 Agent"""

    name = "retrieval"
    description = "负责知识库检索、FAQ 匹配，为用户提供准确的业务信息"
    tools = ["knowledge_search", "faq_lookup"]

    async def run(self, state: CSAssistantState) -> CSAssistantState:
        """执行检索逻辑。"""
        user_query = state.get("user_query", "")
        retrieval_queries = state.get("retrieval_queries", [user_query])
        all_results = []

        for query in retrieval_queries:
            # 先查 FAQ
            faq_result = self.execute_tool("faq_lookup", {"query": query, "limit": 3})
            faq_data = json.loads(faq_result) if isinstance(faq_result, str) else faq_result
            if faq_data.get("success") and faq_data.get("results"):
                all_results.extend(faq_data["results"])

            # 再查知识库
            kb_result = self.execute_tool("knowledge_search", {"query": query, "limit": 5})
            kb_data = json.loads(kb_result) if isinstance(kb_result, str) else kb_result
            if kb_data.get("success") and kb_data.get("results"):
                all_results.extend(kb_data["results"])

        # 去重
        seen = set()
        unique_results = []
        for r in all_results:
            key = r.get("question") or r.get("title") or r.get("content", "")[:50]
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return {
            **state,
            "retrieval_results": unique_results,
            "current_step": WorkflowStep.ANSWERING,
        }
