"""查询智能体

处理查询类意图：查套餐、查账单、查流量、查知识等。
只使用 knowledge + memory 工具，不做写操作。
"""

from agents.base_react_agent import BaseReactSubAgent


class QueryAgent(BaseReactSubAgent):
    name = "query"
    description = "查询智能体：处理信息查询类问题，只读操作"

    allowed_toolsets = ["knowledge", "memory"]

    # 查询类任务用较低温度，确保准确性
    temperature = 0.2
    max_tokens = 2048
    max_iterations = 10

    agent_system_prefix = (
        "你是一个查询客服专员。你的职责是帮助用户查询信息（套餐、账单、流量等）。\n"
        "请使用知识库工具查找准确信息来回答用户。如果查不到，坦诚告知并建议联系人工客服。\n"
        "注意：你只有查询权限，无法办理业务或处理投诉。"
    )
