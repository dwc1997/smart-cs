"""知识库检索工具

基于 SQLite FTS5 的知识库检索和 FAQ 匹配。
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

from tools.registry import registry

logger = logging.getLogger(__name__)

KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", "data/knowledge.db")


def _get_knowledge_db():
    """获取知识库数据库连接。"""
    db_path = KNOWLEDGE_DB_PATH
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_knowledge_db():
    """初始化知识库数据库（如果不存在）。"""
    db_path = KNOWLEDGE_DB_PATH
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            title TEXT,
            content TEXT,
            tags TEXT,
            created_at REAL DEFAULT (strftime('%s', 'now')),
            updated_at REAL DEFAULT (strftime('%s', 'now'))
        );
        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT,
            keywords TEXT,
            hit_count INTEGER DEFAULT 0,
            created_at REAL DEFAULT (strftime('%s', 'now'))
        );
    """)

    # 检查 FTS 表是否存在
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_fts'"
    )
    if cursor.fetchone() is None:
        # 检测 trigram 支持
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _t_fts_test USING fts5(content, tokenize='trigram')")
            conn.execute("DROP TABLE _t_fts_test")
            tokenize_opt = "trigram"
        except sqlite3.OperationalError:
            tokenize_opt = "unicode61"

        conn.execute(
            f"CREATE VIRTUAL TABLE knowledge_fts USING fts5(title, content, tags, tokenize='{tokenize_opt}')"
        )
        # 创建触发器
        conn.executescript("""
            CREATE TRIGGER knowledge_fts_insert AFTER INSERT ON knowledge_base BEGIN
                INSERT INTO knowledge_fts(rowid, title, content, tags) VALUES (
                    new.id, new.title, new.content, new.tags
                );
            END;
            CREATE TRIGGER knowledge_fts_delete AFTER DELETE ON knowledge_base BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, tags) VALUES (
                    'delete', old.id, old.title, old.content, old.tags
                );
            END;
            CREATE TRIGGER knowledge_fts_update AFTER UPDATE ON knowledge_base BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, tags) VALUES (
                    'delete', old.id, old.title, old.content, old.tags
                );
                INSERT INTO knowledge_fts(rowid, title, content, tags) VALUES (
                    new.id, new.title, new.content, new.tags
                );
            END;
        """)

    # 插入示例数据（如果表为空）
    cursor = conn.execute("SELECT COUNT(*) FROM knowledge_base")
    if cursor.fetchone()[0] == 0:
        _insert_sample_data(conn)

    conn.commit()
    conn.close()


def _insert_sample_data(conn):
    """插入示例知识库数据。"""
    sample_knowledge = [
        ("套餐", "5G 畅享套餐", "5G 畅享套餐包含 30GB 流量和 500 分钟通话，月费 128 元。超出流量按 5 元/GB 计费，超出通话按 0.15 元/分钟计费。", "5G,套餐,流量,通话"),
        ("套餐", "4G 飞享套餐", "4G 飞享套餐包含 10GB 流量和 200 分钟通话，月费 58 元。超出流量按 10 元/GB 计费。", "4G,套餐,流量,通话"),
        ("宽带", "家庭宽带安装", "家庭宽带需要提前预约安装，安装费 100 元，3 个工作日内上门。需要提供身份证和房产证明。", "宽带,安装,预约"),
        ("故障", "网络信号差排查", "网络信号差可能的原因：1. 基站维护 2. 室内信号遮挡 3. SIM卡故障。建议先重启手机，如仍无改善可申请换卡。", "信号,网络,故障,排查"),
        ("业务", "国际漫游开通", "国际漫游可通过 APP、营业厅或拨打 10086 开通。开通需预存 500 元话费。不同国家资费不同。", "国际,漫游,开通"),
        ("账单", "账单查询方式", "查询账单可通过：1. 手机 APP 2. 网上营业厅 3. 发送短信 'ZD' 到 10086 4. 拨打 10086 按语音提示操作。", "账单,查询"),
    ]

    for category, title, content, tags in sample_knowledge:
        conn.execute(
            "INSERT INTO knowledge_base (category, title, content, tags) VALUES (?, ?, ?, ?)",
            (category, title, content, tags),
        )

    sample_faq = [
        ("如何查询话费余额？", "您可以通过以下方式查询话费余额：1. 发送短信 'YE' 到 10086 2. 拨打 10086 按 1 号键 3. 打开手机 APP 查看首页余额显示。", "话费", "话费,余额,查询"),
        ("流量用完了怎么办？", "流量用完后您可以：1. 购买流量加油包（5 元/GB 起）2. 升级到更高档位的套餐 3. 连接 WiFi 使用。建议根据您的使用习惯选择合适的方案。", "流量", "流量,加油包,超出"),
        ("如何取消增值业务？", "取消增值业务的方式：1. 发送短信 '0000' 到 10086 查看并取消 2. 登录手机 APP 在 '我的业务' 中取消 3. 拨打 10086 人工服务取消。", "业务", "取消,增值业务,退订"),
        ("宽带故障怎么报修？", "宽带故障报修方式：1. 拨打 10086 按语音提示选择宽带报修 2. 手机 APP 提交故障工单 3. 联系装维师傅。一般 24 小时内处理。", "宽带", "宽带,故障,报修"),
    ]

    for question, answer, category, keywords in sample_faq:
        conn.execute(
            "INSERT INTO faq (question, answer, category, keywords) VALUES (?, ?, ?, ?)",
            (question, answer, category, keywords),
        )


def knowledge_search_tool(query: str, limit: int = 5) -> Dict[str, Any]:
    """在知识库中搜索相关信息。"""
    conn = _get_knowledge_db()
    if not conn:
        return {"success": False, "error": "知识库未初始化", "results": []}

    try:
        # 先尝试 FTS5 搜索
        try:
            cursor = conn.execute(
                "SELECT k.id, k.category, k.title, k.content, k.tags "
                "FROM knowledge_fts fts "
                "JOIN knowledge_base k ON fts.rowid = k.id "
                "WHERE fts.content MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            )
            results = [
                {
                    "id": row[0], "category": row[1], "title": row[2],
                    "content": row[3], "tags": row[4],
                }
                for row in cursor.fetchall()
            ]
        except sqlite3.OperationalError:
            # FTS 搜索失败，回退到 LIKE
            like_pattern = f"%{query}%"
            cursor = conn.execute(
                "SELECT id, category, title, content, tags FROM knowledge_base "
                "WHERE title LIKE ? OR content LIKE ? OR tags LIKE ? LIMIT ?",
                (like_pattern, like_pattern, like_pattern, limit),
            )
            results = [
                {
                    "id": row[0], "category": row[1], "title": row[2],
                    "content": row[3], "tags": row[4],
                }
                for row in cursor.fetchall()
            ]

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        return {"success": False, "error": str(e), "results": []}
    finally:
        conn.close()


def faq_lookup_tool(query: str, limit: int = 3) -> Dict[str, Any]:
    """精确匹配 FAQ 条目。"""
    conn = _get_knowledge_db()
    if not conn:
        return {"success": False, "error": "知识库未初始化", "results": []}

    try:
        like_pattern = f"%{query}%"
        cursor = conn.execute(
            "SELECT id, question, answer, category, keywords, hit_count FROM faq "
            "WHERE question LIKE ? OR keywords LIKE ? "
            "ORDER BY hit_count DESC LIMIT ?",
            (like_pattern, like_pattern, limit),
        )
        results = [
            {
                "id": row[0], "question": row[1], "answer": row[2],
                "category": row[3], "keywords": row[4], "hit_count": row[5],
            }
            for row in cursor.fetchall()
        ]

        # 更新命中次数
        for r in results:
            conn.execute(
                "UPDATE faq SET hit_count = hit_count + 1 WHERE id = ?",
                (r["id"],),
            )
        conn.commit()

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"FAQ lookup failed: {e}")
        return {"success": False, "error": str(e), "results": []}
    finally:
        conn.close()


# ============================================================
# 注册工具
# ============================================================

KNOWLEDGE_SEARCH_SCHEMA = {
    "name": "knowledge_search",
    "description": "在知识库中搜索相关信息，适用于业务规则、产品说明等",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "limit": {"type": "integer", "description": "返回结果数量", "default": 5},
        },
        "required": ["query"],
    },
}

FAQ_LOOKUP_SCHEMA = {
    "name": "faq_lookup",
    "description": "精确匹配 FAQ 常见问题，适用于有标准答案的常见咨询",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "查询关键词"},
            "limit": {"type": "integer", "description": "返回结果数量", "default": 3},
        },
        "required": ["query"],
    },
}

registry.register(
    name="knowledge_search",
    toolset="knowledge",
    schema=KNOWLEDGE_SEARCH_SCHEMA,
    handler=lambda args, **kw: knowledge_search_tool(
        query=args.get("query", ""),
        limit=args.get("limit", 5),
    ),
    description="知识库全文检索",
)

registry.register(
    name="faq_lookup",
    toolset="knowledge",
    schema=FAQ_LOOKUP_SCHEMA,
    handler=lambda args, **kw: faq_lookup_tool(
        query=args.get("query", ""),
        limit=args.get("limit", 3),
    ),
    description="FAQ 精确匹配",
)
