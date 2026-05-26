"""导入冒烟测试

验证所有模块可以正常导入，无需 API 调用。
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


print("=" * 50)
print("Smart-CS 导入冒烟测试")
print("=" * 50)

# 测试配置导入
print("\n[配置]")
test("config.settings", lambda: __import__("config.settings"))

# 测试状态定义导入
print("\n[状态定义]")
def test_states():
    from state.cs_states import (
        CSAssistantState, IntentType, AgentMode, WorkflowStep,
        IntentAnalysisResult, ToolTask, ToolResult, UserProfile,
    )
    assert IntentType.QUERY == "query"
    assert AgentMode.DIRECT_ANSWER == "direct_answer"
    assert WorkflowStep.START == "start"
test("state.cs_states", test_states)

# 测试 prompt 系统导入
print("\n[Prompt 系统]")
def test_prompt_builder():
    from core.prompt_builder import build_system_prompt, load_prompt_file, build_memory_context_block
    prompt = build_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
test("core.prompt_builder", test_prompt_builder)

# 测试工具注册表导入
print("\n[工具注册表]")
def test_registry():
    from tools.registry import registry, ToolRegistry, ToolEntry
    assert isinstance(registry, ToolRegistry)
test("tools.registry", test_registry)

# 测试工具模块导入
print("\n[工具模块]")
def test_knowledge_tools():
    from tools.knowledge_tools import knowledge_search_tool, faq_lookup_tool
test("tools.knowledge_tools", test_knowledge_tools)

def test_ticket_tools():
    from tools.ticket_tools import create_ticket_tool, update_ticket_tool, query_ticket_tool
test("tools.ticket_tools", test_ticket_tools)

def test_calculation_tools():
    from tools.calculation_tools import calculator_tool, billing_query_tool
    result = calculator_tool("2 + 3 * 4")
    assert result["success"] == True
    assert result["result"] == 14
test("tools.calculation_tools", test_calculation_tools)

def test_clarification_tools():
    from tools.clarification_tools import ask_clarification_tool, build_clarification_prompt
    result = ask_clarification_tool("选择套餐", options=["A", "B"])
    assert result["needs_clarification"] == True
test("tools.clarification_tools", test_clarification_tools)

def test_memory_tools():
    from tools.memory_tools import save_memory_tool, recall_memory_tool, search_history_tool
test("tools.memory_tools", test_memory_tools)

# 测试工具注册
print("\n[工具注册]")
def test_tools_registered():
    from tools import registry as reg
    tools = reg.list_tools()
    assert len(tools) >= 11, f"Expected >= 11 tools, got {len(tools)}"
    names = [t["name"] for t in tools]
    assert "knowledge_search" in names
    assert "calculator" in names
    assert "create_ticket" in names
    assert "save_memory" in names
    assert "ask_clarification" in names
test("tools registration", test_tools_registered)

# 测试核心模块导入
print("\n[核心模块]")
def test_session_db():
    from core.session_db import SessionDB, _contains_cjk, _count_cjk
    assert _contains_cjk("你好") == True
    assert _contains_cjk("hello") == False
    assert _count_cjk("你好world") == 2
test("core.session_db", test_session_db)

def test_memory_manager():
    from core.memory_manager import MemoryManager, memory_manager
    assert isinstance(memory_manager, MemoryManager)
test("core.memory_manager", test_memory_manager)

def test_context_compressor():
    from core.context_compressor import (
        estimate_tokens, prune_tool_results, find_compression_boundary,
        compress_messages, sanitize_tool_pairs, should_compress,
    )
    tokens = estimate_tokens("你好世界 hello world")
    assert tokens > 0
test("core.context_compressor", test_context_compressor)

# 测试 Agent 导入
print("\n[Agent 模块]")
def test_agents():
    from agents import (
        BaseAgent, CustomerServiceAgent,
        RetrievalAgent, CalculationAgent, KnowledgeGraphAgent,
    )
    assert CustomerServiceAgent.name == "customer_service"
    assert RetrievalAgent.name == "retrieval"
test("agents", test_agents)

# 测试 ReAct Agent 导入
print("\n[ReAct Agent]")
def test_react_agent():
    from core.react_agent import ReActAgent
    agent = ReActAgent(max_iterations=5)
    assert agent.max_iterations == 5
test("core.react_agent", test_react_agent)

# 测试 LangGraph 图导入
print("\n[LangGraph 图]")
def test_cs_graph():
    from graphs.cs_graph import cs_graph, create_cs_graph
    assert cs_graph is not None
test("graphs.cs_graph", test_cs_graph)

# 测试 API 模型导入
print("\n[API 模型]")
def test_api_models():
    from api.models import QueryRequest, QueryResponse, ResumeRequest, HealthResponse
    req = QueryRequest(query="测试")
    assert req.query == "测试"
test("api.models", test_api_models)

# 测试计算器工具
print("\n[计算器工具验证]")
from tools.calculation_tools import calculator_tool as _calc

def test_calculator_add():
    result = _calc("1 + 2")
    assert result["result"] == 3
test("calculator: 1+2", test_calculator_add)

def test_calculator_complex():
    result = _calc("(10 + 5) * 2 - 3")
    assert result["result"] == 27
test("calculator: (10+5)*2-3", test_calculator_complex)

def test_calculator_division():
    result = _calc("100 / 3")
    assert abs(result["result"] - 33.33333333) < 0.001
test("calculator: 100/3", test_calculator_division)


print("\n" + "=" * 50)
print(f"结果: {passed} 通过, {failed} 失败")
print("=" * 50)

sys.exit(0 if failed == 0 else 1)
