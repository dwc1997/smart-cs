"""本地控制台调试入口

交互式问答控制台，支持流式输出和实时调试。
统一通过 CSEngine 执行（意图分类 → 路由 → ReAct）。
"""

import asyncio
import json
import logging
import os
import sys
import uuid

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config.settings import OPENAI_API_KEY
from core.cs_engine import cs_engine
from core.memory_manager import memory_manager
from core.prompt_builder import build_system_prompt
from tools.registry import registry


def print_banner():
    print("=" * 60)
    print("  Smart-CS 智能客服调试控制台")
    print("  统一执行引擎：意图分类 → 路由 → ReAct")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'tools' 查看可用工具")
    print("  输入 'memory <user_id>' 查看用户记忆")
    print("=" * 60)
    print()


def print_tools():
    tools = registry.list_tools()
    print(f"\n可用工具 ({len(tools)} 个):")
    for t in tools:
        print(f"  [{t['toolset']}] {t['name']}: {t['description']}")
    print()


async def chat_loop():
    """主对话循环。"""
    print_banner()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("错误: 请先配置 OPENAI_API_KEY")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 填入你的 API Key")
        return

    # 初始化知识库
    try:
        from tools.knowledge_tools import _init_knowledge_db
        _init_knowledge_db()
        print("[知识库已初始化]")
    except Exception as e:
        print(f"[知识库初始化失败: {e}]")

    session_id = f"console-{uuid.uuid4().hex[:8]}"
    user_id = "console-user"

    memory_manager.create_session(session_id, user_id=user_id)
    print(f"会话 ID: {session_id}\n")

    while True:
        try:
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if user_input.lower() == "tools":
            print_tools()
            continue

        if user_input.lower().startswith("memory"):
            parts = user_input.split(maxsplit=1)
            uid = parts[1] if len(parts) > 1 else user_id
            context = memory_manager.recall_user_context(uid)
            print(f"\n用户记忆 ({uid}):\n{context or '（空）'}\n")
            continue

        # 保存用户消息
        memory_manager.add_message(session_id, "user", user_input)

        # 获取用户上下文
        user_context = memory_manager.recall_user_context(user_id)

        # 构建 system prompt
        system_prompt = build_system_prompt(
            memory_snapshot=user_context,
            user_profile=memory_manager.get_frozen_profile(),
        )

        # 注入记忆
        message_content = user_input
        if user_context:
            message_content = f"<memory-context>\n{user_context}\n</memory-context>\n\n{user_input}"

        # 流式执行（通过统一引擎）
        print("\n客服: ", end="", flush=True)
        full_response = ""

        async for event in cs_engine.run_streaming(
            messages=[{"role": "user", "content": message_content}],
            system_prompt=system_prompt,
            user_id=user_id,
        ):
            event_type = event.get("type", "")

            if event_type == "intent_classified":
                intent = event["intent"]
                confidence = event["confidence"]
                summary = event["summary"]
                print(f"\n  [意图: {intent} | 置信度: {confidence:.2f} | {summary}]", flush=True)
                print("客服: ", end="", flush=True)

            elif event_type == "token":
                content = event["content"]
                print(content, end="", flush=True)
                full_response += content

            elif event_type == "tool_call":
                name = event["name"]
                args = event["args"]
                print(f"\n  [调用工具: {name}({json.dumps(args, ensure_ascii=False)[:100]})]", flush=True)

            elif event_type == "tool_result":
                name = event["name"]
                result = event["result"][:200]
                print(f"  [工具结果: {name}] {result}", flush=True)

            elif event_type == "clarification":
                data = event["data"]
                question = data.get("question", "")
                options = data.get("options", [])
                print(f"\n  [需要确认] {question}")
                if options:
                    for i, opt in enumerate(options, 1):
                        print(f"    {i}. {opt}")
                    print(f"    {len(options) + 1}. 其他")

            elif event_type == "final":
                if not full_response:
                    full_response = event.get("content", "")
                    print(full_response, end="", flush=True)
                iterations = event.get("iterations", 0)
                tool_calls = event.get("tool_calls", [])
                if tool_calls:
                    print(f"\n  [工具调用次数: {len(tool_calls)}, 迭代次数: {iterations}]", flush=True)

            elif event_type == "error":
                print(f"\n  [错误: {event['error']}]", flush=True)

        print("\n")

        # 保存助手回答
        if full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    # 结束会话
    memory_manager.end_session(session_id)
    memory_manager.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
