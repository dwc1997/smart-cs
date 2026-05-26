"""SQLite + FTS5 会话存储

参考 hermes-agent 的 SessionDB 模式，实现：
- 会话元数据存储
- 完整对话历史
- 用户画像管理
- FTS5 全文检索（支持中文 trigram）
"""

import json
import logging
import os
import random
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    source TEXT DEFAULT 'api',
    model TEXT,
    started_at REAL,
    ended_at REAL,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    phone TEXT,
    package TEXT,
    preferences TEXT DEFAULT '{}',
    history_summary TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    category TEXT,
    subject TEXT,
    description TEXT,
    status TEXT DEFAULT 'open',
    priority TEXT DEFAULT 'normal',
    created_at REAL,
    updated_at REAL,
    resolved_at REAL,
    metadata TEXT DEFAULT '{}'
);
"""

_WRITE_MAX_RETRIES = 15
_WRITE_RETRY_MIN_S = 0.020
_WRITE_RETRY_MAX_S = 0.150


def _contains_cjk(text: str) -> bool:
    """检测文本是否包含 CJK 字符。"""
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F):
            return True
    return False


def _count_cjk(text: str) -> int:
    """统计 CJK 字符数量。"""
    count = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F):
            count += 1
    return count


class SessionDB:
    """SQLite + FTS5 会话数据库"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.RLock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()
        self._init_fts()

    def _init_schema(self):
        """初始化数据库表结构。"""
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def _supports_trigram(self) -> bool:
        """检测 SQLite 是否支持 trigram tokenizer（需要 3.34.0+）。"""
        try:
            test_conn = sqlite3.connect(":memory:")
            test_conn.execute("CREATE VIRTUAL TABLE _t USING fts5(content, tokenize='trigram')")
            test_conn.close()
            return True
        except sqlite3.OperationalError:
            return False

    def _init_fts(self):
        """初始化 FTS5 全文检索表。"""
        self._has_trigram = self._supports_trigram()

        with self._lock:
            # 检查 FTS 表是否已存在
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            )
            if cursor.fetchone() is None:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE messages_fts USING fts5(content, tokenize='unicode61')"
                )

                if self._has_trigram:
                    self._conn.execute(
                        "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram')"
                    )

                # 创建触发器同步数据
                trigram_insert = ""
                trigram_delete = ""
                if self._has_trigram:
                    trigram_insert = """
                        INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                            new.id,
                            COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
                        );
                    """
                    trigram_delete = """
                        INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content) VALUES ('delete', old.id,
                            COALESCE(old.content, '') || ' ' || COALESCE(old.tool_name, '') || ' ' || COALESCE(old.tool_calls, '')
                        );
                    """

                self._conn.executescript(f"""
                    CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content) VALUES (
                            new.id,
                            COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
                        );
                        {trigram_insert}
                    END;

                    CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id,
                            COALESCE(old.content, '') || ' ' || COALESCE(old.tool_name, '') || ' ' || COALESCE(old.tool_calls, '')
                        );
                        {trigram_delete}
                    END;

                    CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id,
                            COALESCE(old.content, '') || ' ' || COALESCE(old.tool_name, '') || ' ' || COALESCE(old.tool_calls, '')
                        );
                        INSERT INTO messages_fts(rowid, content) VALUES (
                            new.id,
                            COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
                        );
                        {trigram_delete}
                        {trigram_insert}
                    END;
                """)
                self._conn.commit()
            else:
                # 表已存在，检查是否有 trigram 表
                cursor = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts_trigram'"
                )
                self._has_trigram = cursor.fetchone() is not None

    def _execute_write(self, fn):
        """带重试的写操作执行器。"""
        for attempt in range(_WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                        return result
                    except BaseException:
                        self._conn.rollback()
                        raise
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower():
                    jitter = random.uniform(_WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S)
                    time.sleep(jitter)
                    continue
                raise
        raise sqlite3.OperationalError("Database write failed after max retries")

    # ============================================================
    # 会话管理
    # ============================================================

    def create_session(
        self, session_id: str, user_id: str = None, source: str = "api", model: str = None
    ) -> None:
        """创建新会话。"""
        def _do(conn):
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, user_id, source, model, started_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, source, model, time.time()),
            )
        self._execute_write(_do)

    def end_session(self, session_id: str) -> None:
        """结束会话。"""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
        self._execute_write(_do)

    # ============================================================
    # 消息管理
    # ============================================================

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: str = None,
        tool_name: str = None,
        metadata: dict = None,
    ) -> int:
        """添加消息到会话。"""
        def _do(conn):
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, role, content, tool_calls, tool_name,
                    time.time(), json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (session_id,),
            )
            return cursor.lastrowid
        return self._execute_write(_do)

    def get_messages(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取会话消息历史。"""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, role, content, tool_calls, tool_name, timestamp, metadata "
                "FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            )
            return [
                {
                    "id": row[0], "role": row[1], "content": row[2],
                    "tool_calls": row[3], "tool_name": row[4],
                    "timestamp": row[5], "metadata": json.loads(row[6] or "{}"),
                }
                for row in cursor.fetchall()
            ]

    # ============================================================
    # 用户画像
    # ============================================================

    def save_user_profile(
        self,
        user_id: str,
        name: str = None,
        phone: str = None,
        package: str = None,
        preferences: dict = None,
        history_summary: str = None,
    ) -> None:
        """保存/更新用户画像。"""
        def _do(conn):
            conn.execute(
                "INSERT INTO user_profiles (user_id, name, phone, package, preferences, history_summary, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "name=COALESCE(excluded.name, name), "
                "phone=COALESCE(excluded.phone, phone), "
                "package=COALESCE(excluded.package, package), "
                "preferences=CASE WHEN excluded.preferences != '{}' THEN excluded.preferences ELSE preferences END, "
                "history_summary=COALESCE(excluded.history_summary, history_summary), "
                "updated_at=excluded.updated_at",
                (
                    user_id, name, phone, package,
                    json.dumps(preferences or {}, ensure_ascii=False),
                    history_summary, time.time(),
                ),
            )
        self._execute_write(_do)

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像。"""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT user_id, name, phone, package, preferences, history_summary, updated_at "
                "FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "user_id": row[0], "name": row[1], "phone": row[2],
                "package": row[3], "preferences": json.loads(row[4] or "{}"),
                "history_summary": row[5], "updated_at": row[6],
            }

    # ============================================================
    # 工单管理
    # ============================================================

    def create_ticket(
        self,
        ticket_id: str,
        user_id: str,
        session_id: str,
        category: str,
        subject: str,
        description: str,
        priority: str = "normal",
    ) -> None:
        """创建工单。"""
        def _do(conn):
            conn.execute(
                "INSERT INTO tickets (id, user_id, session_id, category, subject, description, priority, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticket_id, user_id, session_id, category, subject, description, priority, time.time(), time.time()),
            )
        self._execute_write(_do)

    def update_ticket(self, ticket_id: str, status: str = None, metadata: dict = None) -> None:
        """更新工单状态。"""
        def _do(conn):
            updates = ["updated_at = ?"]
            params = [time.time()]
            if status:
                updates.append("status = ?")
                params.append(status)
                if status == "resolved":
                    updates.append("resolved_at = ?")
                    params.append(time.time())
            if metadata:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata, ensure_ascii=False))
            params.append(ticket_id)
            conn.execute(
                f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        self._execute_write(_do)

    def get_user_tickets(
        self, user_id: str, status: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取用户的工单列表。"""
        with self._lock:
            query = "SELECT id, category, subject, description, status, priority, created_at, updated_at, resolved_at FROM tickets WHERE user_id = ?"
            params = [user_id]
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, params)
            return [
                {
                    "id": row[0], "category": row[1], "subject": row[2],
                    "description": row[3], "status": row[4], "priority": row[5],
                    "created_at": row[6], "updated_at": row[7], "resolved_at": row[8],
                }
                for row in cursor.fetchall()
            ]

    def get_open_tickets(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的未解决工单。"""
        return self.get_user_tickets(user_id, status="open")

    # ============================================================
    # 全文检索
    # ============================================================

    def _sanitize_fts5_query(self, query: str) -> str:
        """清理 FTS5 查询字符串，防止语法错误。"""
        import re
        # 保留引号内的短语
        cleaned = re.sub(r'[+{}()"^]', '', query)
        cleaned = cleaned.strip()
        if not cleaned:
            return '""'
        # 对包含连字符或点号的词加引号
        tokens = cleaned.split()
        result = []
        for token in tokens:
            if '-' in token or '.' in token:
                result.append(f'"{token}"')
            else:
                result.append(token)
        return ' '.join(result)

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """全文搜索消息历史。支持中英文。"""
        if not query.strip():
            return []

        sanitized = self._sanitize_fts5_query(query)

        with self._lock:
            is_cjk = _contains_cjk(query)

            if is_cjk and self._has_trigram:
                cjk_count = _count_cjk(query)
                if cjk_count >= 3:
                    # 使用 trigram FTS 表
                    cursor = self._conn.execute(
                        "SELECT m.id, m.session_id, m.role, m.content, m.timestamp "
                        "FROM messages_fts_trigram fts "
                        "JOIN messages m ON fts.rowid = m.id "
                        "WHERE fts.content MATCH ? ORDER BY rank LIMIT ?",
                        (sanitized, limit),
                    )
                else:
                    # CJK 字符太少，用 LIKE 模糊匹配
                    like_pattern = f"%{query}%"
                    cursor = self._conn.execute(
                        "SELECT id, session_id, role, content, timestamp "
                        "FROM messages WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                        (like_pattern, limit),
                    )
            elif is_cjk:
                # 无 trigram 支持时，CJK 统一用 LIKE
                like_pattern = f"%{query}%"
                cursor = self._conn.execute(
                    "SELECT id, session_id, role, content, timestamp "
                    "FROM messages WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                    (like_pattern, limit),
                )
            else:
                # 非 CJK 使用标准 FTS 表
                cursor = self._conn.execute(
                    "SELECT m.id, m.session_id, m.role, m.content, m.timestamp "
                    "FROM messages_fts fts "
                    "JOIN messages m ON fts.rowid = m.id "
                    "WHERE fts.content MATCH ? ORDER BY rank LIMIT ?",
                    (sanitized, limit),
                )

            return [
                {
                    "id": row[0], "session_id": row[1], "role": row[2],
                    "content": row[3], "timestamp": row[4],
                }
                for row in cursor.fetchall()
            ]

    def close(self):
        """关闭数据库连接。"""
        with self._lock:
            self._conn.close()
