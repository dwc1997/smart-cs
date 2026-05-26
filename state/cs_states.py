"""智能客服 Agent 状态定义

使用 TypedDict 定义 LangGraph StateGraph 的状态结构。
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from enum import Enum

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


# ============================================================
# 枚举定义
# ============================================================

class IntentType(str, Enum):
    """用户意图类型"""
    QUERY = "query"                  # 业务查询（话费、流量、套餐）
    COMPLAINT = "complaint"          # 投诉
    TECHNICAL = "technical"          # 技术支持（网络故障、设备问题）
    BUSINESS = "business"            # 业务办理（开通/取消服务）
    BILLING = "billing"              # 账单争议
    GENERAL = "general"              # 一般咨询
    HUMAN_TRANSFER = "human_transfer"  # 转人工


class AgentMode(str, Enum):
    """Agent 工作模式"""
    DIRECT_ANSWER = "direct_answer"      # 直接回答
    KNOWLEDGE_SEARCH = "knowledge_search"  # 知识库检索
    TICKET = "ticket"                    # 工单处理
    CALCULATION = "calculation"          # 计算处理
    CLARIFICATION = "clarification"      # 主动问询


class WorkflowStep(str, Enum):
    """工作流步骤"""
    START = "start"
    CLASSIFYING = "classifying"
    CONFIRMING = "confirming"
    ROUTING = "routing"
    SEARCHING = "searching"
    PROCESSING = "processing"
    ANSWERING = "answering"
    COMPLETED = "completed"
    ERROR = "error"


# ============================================================
# 子结构定义
# ============================================================

class IntentAnalysisResult(TypedDict, total=False):
    """意图分析结果"""
    intent_type: str           # IntentType 值
    confidence: float          # 置信度 0-1
    summary: str               # 用户需求摘要
    entities: Dict[str, Any]   # 提取的实体（手机号、套餐名等）
    requires_clarification: bool  # 是否需要确认


class ToolTask(TypedDict, total=False):
    """工具任务定义"""
    tool_name: str             # 工具名称
    arguments: Dict[str, Any]  # 工具参数
    description: str           # 任务描述


class ToolResult(TypedDict, total=False):
    """工具执行结果"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str]


class UserProfile(TypedDict, total=False):
    """用户画像"""
    user_id: str
    name: Optional[str]
    phone: Optional[str]
    package: Optional[str]       # 当前套餐
    preferences: Dict[str, Any]  # 用户偏好
    history_summary: Optional[str]


# ============================================================
# 主状态定义
# ============================================================

class CSAssistantState(TypedDict, total=False):
    """智能客服 Agent 主状态"""

    # 消息历史（自动累加）
    messages: Annotated[List[AnyMessage], add_messages]

    # 用户输入
    user_query: str                # 处理后的用户查询
    original_input: str            # 原始用户输入
    session_id: str                # 会话 ID
    user_id: Optional[str]         # 用户 ID

    # 意图分析
    intent_analysis: Optional[IntentAnalysisResult]
    intent_confirmed: bool         # 意图是否已确认
    confirmed_intent: Optional[str]  # 确认后的意图类型

    # 工作流
    current_mode: str              # AgentMode 值
    current_step: str              # WorkflowStep 值
    error_messages: List[str]

    # 多智能体调度
    assigned_agent: Optional[str]  # 分配的子 Agent 名称

    # 工具调用
    planned_tasks: List[ToolTask]
    tool_execution_results: List[ToolResult]

    # 记忆
    user_profile: Optional[UserProfile]
    memory_context: Optional[str]  # 召回的记忆上下文
    short_term_memory: List[Dict[str, Any]]

    # 知识检索
    retrieval_queries: List[str]
    retrieval_results: List[Dict[str, Any]]

    # 元数据
    metadata: Dict[str, Any]
