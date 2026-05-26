"""图谱 Agent（TODO）

负责知识图谱查询。
当前为占位实现，后续接入知识图谱后完善。
"""

import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from state.cs_states import CSAssistantState, WorkflowStep

logger = logging.getLogger(__name__)


class KnowledgeGraphAgent(BaseAgent):
    """知识图谱 Agent（TODO）"""

    name = "knowledge_graph"
    description = "负责知识图谱查询，提供结构化的关联信息（待实现）"
    tools = []  # TODO: 接入知识图谱工具

    async def run(self, state: CSAssistantState) -> CSAssistantState:
        """TODO: 接入知识图谱后实现。"""
        logger.warning("KnowledgeGraphAgent is not yet implemented")

        return {
            **state,
            "retrieval_results": [
                {"source": "knowledge_graph", "content": "知识图谱功能暂未实现，请使用知识库检索。"}
            ],
            "current_step": WorkflowStep.ANSWERING,
        }
