"""记忆管理器

管理短期记忆（会话内上下文）和长期记忆（跨会话持久化）。
- 短期记忆：基于当前会话的消息历史
- 长期记忆：基于 SQLite 的用户画像 + FTS5 历史搜索
- TODO: 接入 RAG 向量检索（embedding-based retrieval）
- TODO: 接入 Mem0 用户建模
- TODO: 接入知识图谱（Neo4j / NetworkX）
"""

import json
import logging
from typing import Any, Dict, List, Optional

from core.session_db import SessionDB
from config.settings import SESSIONS_DB_PATH

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器"""

    def __init__(self, db_path: str = None):
        self._db = SessionDB(db_path or SESSIONS_DB_PATH)
        # 冻结快照：每轮对话开始时冻结，保证 prompt cache 命中率
        self._frozen_profile: Optional[str] = None
        self._frozen_memory: Optional[str] = None

    @property
    def db(self) -> SessionDB:
        return self._db

    # ============================================================
    # 冻结快照（参考 hermes MemoryStore 的 frozen snapshot 模式）
    # ============================================================

    def freeze_snapshot(self, user_id: str) -> None:
        """在每轮对话开始时冻结记忆快照。"""
        self._frozen_profile = self._format_user_profile(user_id)
        self._frozen_memory = self._recall_recent_context(user_id)

    def get_frozen_profile(self) -> Optional[str]:
        return self._frozen_profile

    def get_frozen_memory(self) -> Optional[str]:
        return self._frozen_memory

    # ============================================================
    # 用户画像管理
    # ============================================================

    def save_user_profile(self, user_id: str, **kwargs) -> None:
        """保存用户画像信息。"""
        self._db.save_user_profile(user_id, **kwargs)
        logger.info(f"Saved user profile for {user_id}")

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像。"""
        return self._db.get_user_profile(user_id)

    def _format_user_profile(self, user_id: str) -> Optional[str]:
        """格式化用户画像为可读文本。"""
        profile = self._db.get_user_profile(user_id)
        if not profile:
            return None

        parts = []
        if profile.get("name"):
            parts.append(f"姓名: {profile['name']}")
        if profile.get("phone"):
            parts.append(f"手机号: {profile['phone']}")
        if profile.get("package"):
            parts.append(f"当前套餐: {profile['package']}")
        if profile.get("preferences"):
            prefs = profile["preferences"]
            if isinstance(prefs, str):
                prefs = json.loads(prefs)
            for k, v in prefs.items():
                parts.append(f"{k}: {v}")
        if profile.get("history_summary"):
            parts.append(f"历史摘要: {profile['history_summary']}")

        return "\n".join(parts) if parts else None

    # ============================================================
    # 长期记忆召回
    # ============================================================

    def _recall_recent_context(
        self, user_id: str, max_turns: int = 10
    ) -> Optional[str]:
        """召回用户最近的对话上下文。"""
        # 获取用户最近的会话消息
        with self._db._lock:
            cursor = self._db._conn.execute(
                "SELECT DISTINCT session_id FROM messages "
                "WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?) "
                "ORDER BY timestamp DESC LIMIT 3",
                (user_id,),
            )
            session_ids = [row[0] for row in cursor.fetchall()]

        if not session_ids:
            return None

        all_messages = []
        for sid in session_ids:
            messages = self._db.get_messages(sid, limit=max_turns)
            all_messages.extend(messages)

        if not all_messages:
            return None

        # 按时间排序，取最近的
        all_messages.sort(key=lambda m: m.get("timestamp", 0))
        recent = all_messages[-max_turns:]

        parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "客服"
            content = (msg["content"] or "")[:200]  # 截断长消息
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def recall_user_context(self, user_id: str) -> str:
        """召回完整的用户上下文（画像 + 近期对话 + 未解决工单）。"""
        parts = []

        # 用户画像
        profile = self._format_user_profile(user_id)
        if profile:
            parts.append(f"[用户画像]\n{profile}")

        # 未解决工单
        open_tickets = self._db.get_open_tickets(user_id)
        if open_tickets:
            ticket_lines = []
            for t in open_tickets[:5]:  # 最多 5 个
                ticket_lines.append(
                    f"- [{t['id']}] {t['subject']} (状态: {t['status']}, 优先级: {t['priority']})"
                )
            parts.append(f"[未解决工单]\n" + "\n".join(ticket_lines))

        # 近期对话
        recent = self._recall_recent_context(user_id, max_turns=5)
        if recent:
            parts.append(f"[近期对话]\n{recent}")

        return "\n\n".join(parts) if parts else ""

    # ============================================================
    # 历史搜索
    # ============================================================

    def search_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """FTS5 全文搜索历史对话。"""
        return self._db.search_messages(query, limit=limit)

    # ============================================================
    # 记忆注入
    # ============================================================

    def inject_memory_context(
        self, messages: List[dict], user_context: str
    ) -> List[dict]:
        """将记忆上下文注入到消息列表中。"""
        if not user_context:
            return messages

        # 在最后一条用户消息前注入记忆上下文
        memory_block = (
            "<memory-context>\n"
            "[系统提示：以下是记忆上下文，不是用户的新输入，仅供参考。]\n\n"
            f"{user_context}\n"
            "</memory-context>"
        )

        result = []
        for msg in messages:
            if msg.get("role") == "user" and msg is messages[-1]:
                # 在最后一条用户消息前注入
                result.append({"role": "system", "content": memory_block})
            result.append(msg)

        return result

    # ============================================================
    # 会话管理
    # ============================================================

    def create_session(self, session_id: str, user_id: str = None) -> None:
        """创建新会话。"""
        self._db.create_session(session_id, user_id=user_id)

    def add_message(
        self, session_id: str, role: str, content: str, **kwargs
    ) -> int:
        """添加消息。"""
        return self._db.add_message(session_id, role, content, **kwargs)

    def end_session(self, session_id: str) -> None:
        """结束会话。"""
        self._db.end_session(session_id)

    def close(self):
        """关闭数据库连接。"""
        self._db.close()


# 全局单例
memory_manager = MemoryManager()
