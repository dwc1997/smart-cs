"""费用智能体

处理费用类意图：话费、扣费、充值、退款、费用计算等。
使用 knowledge + calculation + memory 工具。
"""

from agents.base_react_agent import BaseReactSubAgent


class BillingAgent(BaseReactSubAgent):
    name = "billing"
    description = "费用智能体：处理费用查询和计算相关问题"

    allowed_toolsets = ["knowledge", "calculation", "memory"]

    # 费用计算需要精确，低温度
    temperature = 0.1
    max_tokens = 2048
    max_iterations = 12

    agent_system_prefix = (
        "你是一个费用处理专员。你的职责是处理用户关于费用的问题：\n"
        "1. 使用知识库查找资费标准、优惠政策等信息\n"
        "2. 使用计算工具进行精确的费用计算（如套餐对比、话费估算等）\n"
        "3. 清晰地向用户解释费用构成和计算结果\n"
        "4. 涉及退款等问题时，说明流程和时间\n"
        "注意：费用相关回答必须准确，计算结果需要验证。"
    )
