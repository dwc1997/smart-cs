# Smart-CS 智能客服项目

## 项目概述

基于 LangGraph + LangChain 的智能客服 Agent 系统，支持意图识别、多智能体调度、ReAct 循环、长期记忆管理。

## 技术栈

- **LangGraph** >= 0.6.0 -- StateGraph 工作流
- **LangChain** >= 0.3.0 -- LLM 抽象层
- **FastAPI** -- HTTP API + SSE 流式输出
- **SQLite + FTS5** -- 会话存储与全文检索
- **httpx** -- 直接 HTTP 调用 LLM API

## 项目结构

```
smart-cs/
├── prompts/          # 分层 prompt 文件（AGENT.md, MEMORY.md, TOOLS.md, CUSTOMER_SERVICE.md）
├── config/           # 配置（settings.py）
├── core/             # 核心引擎（prompt_builder, session_db, memory_manager, react_agent, context_compressor）
├── agents/           # 多智能体（customer_service, retrieval, calculation, knowledge_graph）
├── tools/            # 工具注册表 + 工具实现
├── graphs/           # LangGraph 图定义
├── state/            # TypedDict 状态定义
├── api/              # FastAPI 应用
├── cli/              # 本地控制台调试
├── data/             # SQLite 数据库文件
└── tests/            # 测试
```

## 关键设计模式

### 分层 Prompt
参考 hermes-agent 的 prompt_builder 模式，将 system prompt 从多个 .md 文件组装：
- `prompts/AGENT.md` -- 客服身份与行为准则
- `prompts/CUSTOMER_SERVICE.md` -- 业务知识与话术规范
- `prompts/MEMORY.md` -- 记忆使用指导
- `prompts/TOOLS.md` -- 工具使用规范

### 工具注册表
参考 hermes-agent 的 ToolRegistry 模式，支持：
- 工具分组（toolset）
- 可用性检查（check_fn with TTL cache）
- 自动 schema 发现

### 记忆系统
- 短期记忆：LangGraph checkpointer 管理会话内状态
- 长期记忆：SQLite + FTS5 全文检索，支持中文 trigram
- 冻结快照：每轮对话前冻结 memory，保证 prompt cache 命中率
- TODO: RAG 向量检索、Mem0 用户建模、知识图谱

### ReAct 循环
思考(LLM) → 行动(工具调用) → 观察(工具结果) → 迭代
- 最大迭代数中止（默认 15 轮）
- 重复检测（连续 3 次相同调用则跳出）
- 主动问询（工具返回 needs_clarification 时触发）

## 本地调试

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 3. 控制台调试
python cli/console.py

# 4. 启动 API 服务
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. 运行测试
python tests/test_simple.py
python tests/test_memory.py
```

## 添加新工具

1. 在 `tools/` 下创建工具文件
2. 实现工具函数（返回 dict 或 str）
3. 定义 OpenAI function calling schema
4. 调用 `registry.register()` 注册
5. 在 `tools/__init__.py` 中导入模块

## 添加新 Agent

1. 在 `agents/` 下创建 Agent 文件
2. 继承 `BaseAgent`，实现 `run()` 方法
3. 在 `agents/customer_service_agent.py` 中添加路由逻辑
