# Smart-CS — 生产级智能客服 Agent 系统

> 一个以**深度技术广度**为目标的智能客服 Agent 系统，覆盖主流 Agent 工程技术栈，参考 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 架构设计。

## 🎯 项目定位

本项目不是又一个"套壳 LLM 的客服机器人"，而是一个**Agent 技术全景实战平台**：

- ✅ 已实现的技术用 ✅ 标记，可直接在面试中展开讲解
- 🚧 TODO 中的技术是明确的演进方向，体现技术视野

**目标岗位**：Agent 开发工程师 / LLM 应用架构师

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│              FastAPI (SSE Streaming) + CLI Console                │
├─────────────────────────────────────────────────────────────────┤
│                      Agent Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Intent Router │  │  Planner     │  │ Multi-Agent Dispatch  │  │
│  │ (意图识别)    │  │  (规划器)     │  │ (子Agent编排)         │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
│         │                 │                       │              │
│  ┌──────▼─────────────────▼───────────────────────▼───────────┐  │
│  │                  ReAct Agent Loop                           │  │
│  │         think → act → observe → iterate (max 15轮)          │  │
│  │         + 重复检测 + 主动问询 + 错误恢复                     │  │
│  └──────────────────────┬────────────────────────────────────┘  │
├─────────────────────────┼───────────────────────────────────────┤
│                    Tool Layer                                    │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │ Knowledge   │ │ Ticket   │ │ Calc     │ │ Clarification   │  │
│  │ (知识检索)  │ │ (工单)   │ │ (计费)   │ │ (推荐问/澄清)   │  │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────────┘  │
│  ┌────────────┐ ┌────────────────────────────────────────────┐  │
│  │ Memory     │ │           Tool Registry                     │  │
│  │ (记忆工具) │ │  (自注册 + TTL缓存 + Shadow保护 + Schema)   │  │
│  └────────────┘ └────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Memory Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Short-term    │  │ Long-term    │  │ Context Compressor    │  │
│  │ (冻结快照)    │  │ (SQLite+FTS5)│  │ (上下文压缩)          │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Prompt Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ AGENT.md     │  │ CS.md        │  │ MEMORY.md + TOOLS.md  │  │
│  │ (身份/行为)  │  │ (业务知识)   │  │ (记忆指导/工具规范)   │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Storage Layer                                 │
│         SQLite + WAL + FTS5 (trigram CJK) + Retry Logic         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
smart-cs/
├── README.md                # 本文件
├── CLAUDE.md                # AI 辅助开发指南
├── .env.example             # 环境变量模板
├── requirements.txt         # Python 依赖
│
├── config/                  # 配置层
│   └── settings.py          # LLM/DB/Agent 参数
│
├── state/                   # 状态定义
│   └── cs_states.py         # LangGraph TypedDict 状态机
│
├── prompts/                 # 分层 Prompt（参考 Hermes SOUL.md 模式）
│   ├── AGENT.md             # Agent 身份 + 行为准则
│   ├── CUSTOMER_SERVICE.md  # 业务知识 + 话术规范
│   └── TOOLS.md             # 工具使用规范
│
├── core/                    # 核心引擎
│   ├── prompt_builder.py    # 5层 Prompt 组装器
│   ├── session_db.py        # SQLite + FTS5 会话存储
│   ├── memory_manager.py    # 冻结快照 + 记忆召回
│   ├── react_agent.py       # ReAct 循环引擎
│   └── context_compressor.py # 上下文压缩工具
│
├── tools/                   # 工具注册表 + 工具实现
│   ├── registry.py          # 全局单例注册表
│   ├── knowledge_tools.py   # FTS5 知识检索 + FAQ
│   ├── ticket_tools.py      # 工单 CRUD
│   ├── calculation_tools.py # AST安全计算器 + 账单查询
│   ├── clarification_tools.py # 推荐问/澄清
│   └── memory_tools.py      # 记忆存储/召回/历史搜索
│
├── agents/                  # 多智能体
│   ├── base_agent.py        # 抽象基类
│   ├── customer_service_agent.py # 主编排 Agent
│   ├── retrieval_agent.py   # 知识检索 Agent
│   ├── calculation_agent.py # 计费计算 Agent
│   └── knowledge_graph_agent.py # 知识图谱 Agent (TODO)
│
├── graphs/                  # LangGraph 图定义
│   └── cs_graph.py          # 意图→路由→执行 工作流
│
├── api/                     # HTTP API
│   ├── main.py              # FastAPI + SSE 流式
│   └── models.py            # Pydantic 模型
│
├── cli/                     # 本地调试
│   └── console.py           # 交互式控制台
│
└── tests/                   # 测试
    ├── test_simple.py       # 导入 + 冒烟测试
    └── test_memory.py       # 记忆系统集成测试
```

---

## ✅ 已实现的技术特性

### 🔄 Agent Loop & ReAct
| 特性 | 状态 | 说明 |
|------|------|------|
| ReAct 循环（think→act→observe） | ✅ | `core/react_agent.py`，支持同步 + 流式两种模式 |
| 最大迭代中止 | ✅ | 默认 15 轮，防止无限循环 |
| 重复调用检测 | ✅ | 连续 3 次相同工具调用自动跳出 |
| 主动问询（Clarify） | ✅ | 工具返回 `needs_clarification` 时触发 |
| 流式输出（SSE） | ✅ | 逐 token + 工具调用/结果事件流 |
| Function Calling | ✅ | OpenAI 格式，自动从注册表生成 schema |

### 🧠 记忆系统
| 特性 | 状态 | 说明 |
|------|------|------|
| 短期记忆（冻结快照） | ✅ | 会话开始时冻结 profile + 近期上下文，保证 prompt cache 命中率 |
| 长期记忆（SQLite + FTS5） | ✅ | 用户画像、会话历史、全文检索，支持中文 trigram |
| 上下文压缩 | ✅ | 实现了压缩函数（尚未集成到主流程） |
| 记忆注入 | ✅ | `<memory-context>` XML 标签注入到 prompt |

### 📝 Prompt 分层
| 特性 | 状态 | 说明 |
|------|------|------|
| 5层 Prompt 组装 | ✅ | 静态 .md + 动态 profile + 工具上下文 |
| 头尾保护截断 | ✅ | 超长 prompt 文件的 70/20 分割策略 |
| 记忆上下文隔离 | ✅ | XML 标签包裹，防止注入污染 |

### 🤖 Multi-Agent
| 特性 | 状态 | 说明 |
|------|------|------|
| 意图识别路由 | ✅ | LLM 分类 → 子Agent分发 |
| 子Agent编排 | ✅ | Retrieval / Calculation / KnowledgeGraph |
| 工单自动创建 | ✅ | 投诉类意图自动建高优先级工单 |

### 🔧 工具系统
| 特性 | 状态 | 说明 |
|------|------|------|
| 工具注册表 | ✅ | 全局单例，TTL 缓存，Shadow 保护 |
| 工具分组（Toolset） | ✅ | knowledge / ticket / calculation / interaction / memory |
| 可用性检查 | ✅ | check_fn + 30s TTL 缓存 |
| AST安全计算器 | ✅ | 白名单运算符，无 eval() |
| 推荐问/澄清 | ✅ | 多选 + 开放式，鲁棒的选项解析 |

### 💾 存储引擎
| 特性 | 状态 | 说明 |
|------|------|------|
| SQLite WAL 模式 | ✅ | 并发读写优化 |
| FTS5 全文检索 | ✅ | unicode61 + trigram 双分词器，自动检测 |
| 写重试机制 | ✅ | 最多 15 次重试 + 随机抖动 |
| FTS5 查询消毒 | ✅ | 特殊字符清理，防止查询注入 |

### 🌐 API & 交互
| 特性 | 状态 | 说明 |
|------|------|------|
| FastAPI SSE 流式 API | ✅ | 逐 token 推送 + 工具事件 |
| 交互式 CLI 控制台 | ✅ | 流式输出 + 工具可视化 |
| CORS 跨域 | ✅ | 全来源放行 |
| LangSmith 集成 | ✅ | 可选的调用链追踪 |

---

## 🚀 TODO — 技术演进路线图

> 以下特性按**面试价值**排序，每项都对应一个 Agent 工程核心技术点。

### 🔴 P0 — 核心架构补全（面试必问）

- [x] **统一执行路径** — 当前存在两条并行路径（LangGraph 图 vs ReActAgent），API/CLI 绕过 Graph 直接调用 ReActAgent。需要统一为 Graph 驱动，ReAct 作为 Graph 内部节点
- [ ] **集成上下文压缩到主流程** — `context_compressor.py` 已实现但从未被调用，需要在 ReAct 循环中自动触发压缩（参考 Hermes 的 ContextEngine 抽象）
- [ ] **LangGraph Checkpointer** — 用 `MemorySaver` 或 `SqliteSaver` 实现真正的 Graph 状态持久化，支持会话恢复
- [ ] **Agent Loop 健壮性** — 参考 Hermes 的 IterationBudget（预算制）+ ErrorClassifier（错误分类 → 重试/降级/中止）+ ToolGuardrails（循环检测 → 熔断）
- [ ] **中断/恢复机制** — API 的 `/resume` 端点返回 501，需要实现真正的 interrupt → human-in-the-loop → resume 流程

### 🟡 P1 — 高级 RAG & 知识（面试加分）

- [ ] **Agentic RAG** — 不只是"检索→拼接→生成"，而是 Agent 自主决定是否检索、检索什么、如何验证结果质量、是否需要二次检索
- [ ] **问题重写（Query Rewriting）** — 用户原始问题 → LLM 改写为更适合检索的查询（HyDE、多查询、子问题分解）
- [ ] **向量检索** — 接入 Embedding 模型（如 text2vec-base-chinese），用 FAISS/Milvus 替代纯 FTS5，支持语义搜索
- [ ] **Reranking** — 检索结果重排序（Cross-Encoder / Cohere Rerank / BGE-Reranker），提升精度
- [ ] **知识图谱** — 从 stub 变为实际功能，用 Neo4j 或 NetworkX 构建实体关系图，支持多跳推理
- [ ] **混合检索** — FTS5 关键词 + 向量语义 + 知识图谱三路召回，RRF 融合排序

### 🟢 P2 — 工程化 & 可观测性（体现工程素养）

- [ ] **Prometheus 指标** — 对话延迟、Token 消耗、工具调用成功率、意图分布、用户满意度
- [ ] **结构化日志** — JSON 格式日志 + trace_id 贯穿全链路（参考 Hermes 的 LangSmith 集成）
- [ ] **Guardrails** — 输入/输出安全检查（PII 脱敏、敏感词过滤、幻觉检测），参考 Hermes 的 prompt injection scanner
- [ ] **A/B 测试框架** — 不同 prompt / 模型 / 检索策略的效果对比
- [ ] **用户反馈闭环** — 👍👎 收集 → 人工标注 → 训练数据 → 模型微调
- [ ] **配置热更新** — prompt / 知识库 / 路由规则的运行时更新，无需重启
- [ ] **健康检查 & 熔断** — LLM 服务不可用时的降级策略（缓存回复 / 转人工）

### 🔵 P3 — 高级 Agent 能力（体现技术深度）

- [ ] **规划器（Planner）** — 复杂任务的步骤分解，参考 Hermes 的 delegate_task + Kanban 分解模式
- [ ] **Sandbox 代码执行** — 用户问计费问题时，Agent 在沙箱中运行 Python 脚本计算（参考 Hermes 的 Docker/Modal 多后端）
- [ ] **多轮记忆压缩** — 参考 Hermes 的 conversation_compression：头尾保护 + 中间摘要 + Resolved/Pending 跟踪
- [ ] **Tool Guardrails** — 工具调用循环检测 + 熔断器（参考 Hermes 的 tool_guardrails.py）
- [ ] **错误分类 & 降级** — 结构化错误分类 → 自动重试 / 切换模型 / 压缩上下文 / 中止（参考 Hermes 的 error_classifier）
- [ ] **Credential Pool** — 多 API Key 轮换 + 冷却期管理（参考 Hermes 的 credential_pool.py）
- [ ] **Human-in-the-Loop 审批** — 敏感操作（退款、销户）需要人工确认才能执行
- [ ] **Prompt Caching** — 参考 Hermes 的 Anthropic `system_and_3` 策略，多轮对话 input token 成本降低 75%

### 🟣 P4 — 生态 & 扩展性（体现架构视野）

- [ ] **Plugin 架构** — 模型/记忆/检索/工具的插件化，参考 Hermes 的 Provider ABC + plugins/ 目录
- [ ] **MCP 集成** — Model Context Protocol 客户端，让 Agent 能连接外部 MCP Server 获取工具
- [ ] **多渠道接入** — Telegram / 微信 / 钉钉 / Slack 适配层
- [ ] **多模型切换** — Provider 抽象层（OpenAI / 通义千问 / DeepSeek / 本地模型），带 failover
- [ ] **Prompt 版本管理** — .md 文件的 Git 版本控制 + 灰度发布
- [ ] **会话导出 & 分析** — ShareGPT 格式导出（参考 Hermes 的 trajectory.py），支持 RLHF 训练
- [ ] **Webhook 事件** — 会话开始/结束/升级/满意度的 Webhook 推送

### ⚪ P5 — 前沿技术（体现技术前瞻性）

- [ ] **Structured Output** — 强制 LLM 输出符合 JSON Schema 的结构化数据（意图、情感、实体），参考 Hermes 的 outlines 集成
- [ ] **语义缓存** — 相似问题命中缓存直接返回（embedding + cosine similarity），降低延迟和成本
- [ ] **多模态输入** — 支持图片（产品截图）、语音（ASR 转文字）输入
- [ ] **评估框架** — 自动化评估：准确性、相关性、幻觉率、响应时间，集成 Weights & Biases
- [ ] **Agent 自优化** — 根据用户反馈自动调整 prompt / 路由规则 / 工具优先级

---

## 🛠️ 快速开始

```bash
# 1. 克隆仓库
git clone git@github.com:dwc1997/smart-cs.git
cd smart-cs

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 4. 启动 API 服务
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. 或使用 CLI 调试
python cli/console.py

# 6. 运行测试
python tests/test_simple.py
python tests/test_memory.py
```

---

## 🧩 设计理念

### 参考 Hermes Agent 的核心设计

| 设计模式 | Hermes 实现 | Smart-CS 对应 |
|---------|------------|---------------|
| 三层 Prompt（stable/context/volatile） | `agent/system_prompt.py` | `core/prompt_builder.py` 5层组装 |
| 自注册工具表 + Toolset | `tools/registry.py` + `toolsets.py` | `tools/registry.py` |
| 冻结快照 | `agent/memory_manager.py` | `core/memory_manager.py` |
| Context Engine 抽象 | `agent/context_engine.py` | `core/context_compressor.py` |
| Error Classifier | `agent/error_classifier.py` | TODO: P0 |
| Tool Guardrails | `agent/tool_guardrails.py` | TODO: P3 |
| 多Agent委派 | `tools/delegate_tool.py` | `agents/` 子Agent编排 |
| Sandbox 执行 | `tools/environments/` | TODO: P3 |

### 与 Hermes 的差异化

Hermes 是一个**通用 AI Agent 平台**，Smart-CS 专注于**垂直领域客服场景**：

- 意图识别 → 路由 → 子Agent 的**有状态工作流**（LangGraph StateGraph）
- 电信领域知识库 + FAQ + 工单系统的**业务集成**
- 面向终端用户的**SSE 流式体验**
- 中文场景的**深度优化**（trigram 分词、CJK token 估算）

---

## 📊 技术栈

| 组件 | 技术选型 |
|------|---------|
| LLM 框架 | LangChain + LangGraph |
| Agent 模式 | ReAct Loop + StateGraph |
| API 框架 | FastAPI + SSE |
| 存储引擎 | SQLite + WAL + FTS5 |
| 全文检索 | FTS5 trigram（中文优化） |
| 流式输出 | AsyncOpenAI + httpx |
| 安全 | AST-based 计算器、FTS5 查询消毒 |

---

## 📈 代码统计

- **总代码量**: ~3,774 行
- **核心模块**: 8 个（config, state, core, tools, agents, graphs, api, cli）
- **工具数量**: 9 个（知识搜索、FAQ、工单 CRUD、计算器、账单查询、澄清、记忆存储/召回/搜索）
- **子Agent**: 4 个（客服编排、知识检索、计费计算、知识图谱）
- **Prompt 文件**: 3 个（AGENT.md, CUSTOMER_SERVICE.md, TOOLS.md）
- **测试覆盖**: 导入冒烟 + 记忆系统集成

---

## 📝 License

MIT
