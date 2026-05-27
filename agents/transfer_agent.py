"""转人工智能体

处理转人工意图。最小工具集，主要负责收集信息并转交人工客服。
"""

from agents.base_react_agent import BaseReactSubAgent


class TransferAgent(BaseReactSubAgent):
    name = "transfer"
    description = "转人工智能体：处理人工转接，收集转接信息"

    allowed_toolsets = ["memory"]

    # 转人工场景用正常温度
    temperature = 0.3
    max_tokens = 1024
    max_iterations = 5

    agent_system_prefix = (
        "你是一个转人工客服专员。用户要求转接人工客服，你需要：\n"
        "1. 确认用户的转接需求\n"
        "2. 简要收集用户问题的摘要，方便人工客服了解情况\n"
        "3. 告知用户正在转接，请稍候\n"
        "4. 回复要简洁，不要拖延用户等待时间\n"
        "你的最终回复应该以「正在为您转接人工客服」开头。"
    )
