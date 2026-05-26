"""工具模块初始化

导入所有工具模块以触发注册。
"""

from tools.registry import registry

# 导入工具模块以触发注册
from tools import knowledge_tools
from tools import ticket_tools
from tools import calculation_tools
from tools import clarification_tools
from tools import memory_tools

__all__ = ["registry"]
