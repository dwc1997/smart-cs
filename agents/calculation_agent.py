"""计算 Agent

负责费用计算、账单查询等计算相关任务。
"""

import json
import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from state.cs_states import CSAssistantState, AgentMode, WorkflowStep

logger = logging.getLogger(__name__)


class CalculationAgent(BaseAgent):
    """计算 Agent"""

    name = "calculation"
    description = "负责费用计算、账单查询、套餐对比等计算相关任务"
    tools = ["calculator", "billing_query"]

    async def run(self, state: CSAssistantState) -> CSAssistantState:
        """执行计算逻辑。"""
        user_query = state.get("user_query", "")
        planned_tasks = state.get("planned_tasks", [])
        results = []

        for task in planned_tasks:
            tool_name = task.get("tool_name", "")
            arguments = task.get("arguments", {})

            if tool_name in self.tools:
                result = self.execute_tool(tool_name, arguments)
                result_data = json.loads(result) if isinstance(result, str) else result
                results.append({
                    "tool_name": tool_name,
                    "success": result_data.get("success", False),
                    "result": result_data,
                })

        # 如果没有预规划的任务，尝试默认的费用查询
        if not results:
            result = self.execute_tool("billing_query", {
                "query_type": "compare_packages",
            })
            result_data = json.loads(result) if isinstance(result, str) else result
            results.append({
                "tool_name": "billing_query",
                "success": result_data.get("success", False),
                "result": result_data,
            })

        return {
            **state,
            "tool_execution_results": results,
            "current_step": WorkflowStep.ANSWERING,
        }
