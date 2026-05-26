"""记忆系统测试"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session_db import SessionDB
from core.memory_manager import MemoryManager

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
print("Smart-CS 记忆系统测试")
print("=" * 50)

# 使用临时数据库
tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmpfile.close()
db_path = tmpfile.name

try:
    # 测试 SessionDB
    print("\n[SessionDB]")
    db = SessionDB(db_path)

    def test_create_session():
        db.create_session("s1", user_id="u1")
        db.create_session("s2", user_id="u2")
    test("create_session", test_create_session)

    def test_add_message():
        db.add_message("s1", "user", "你好")
        db.add_message("s1", "assistant", "您好，请问有什么可以帮您？")
        db.add_message("s1", "user", "我想查话费")
        messages = db.get_messages("s1")
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好"
    test("add_message & get_messages", test_add_message)

    def test_user_profile():
        db.save_user_profile("u1", name="张三", phone="13800138000", package="5G畅享套餐")
        profile = db.get_user_profile("u1")
        assert profile is not None
        assert profile["name"] == "张三"
        assert profile["phone"] == "13800138000"
        assert profile["package"] == "5G畅享套餐"
    test("user_profile", test_user_profile)

    def test_user_profile_update():
        db.save_user_profile("u1", package="4G飞享套餐")
        profile = db.get_user_profile("u1")
        assert profile["package"] == "4G飞享套餐"
        assert profile["name"] == "张三"  # 保留原有字段
    test("user_profile_update", test_user_profile_update)

    def test_tickets():
        db.create_ticket("TK-001", "u1", "s1", "complaint", "网络信号差", "最近信号一直不好")
        db.create_ticket("TK-002", "u1", "s1", "billing", "账单疑问", "上月费用偏高")
        tickets = db.get_user_tickets("u1")
        assert len(tickets) == 2
    test("tickets create & query", test_tickets)

    def test_ticket_update():
        db.update_ticket("TK-001", status="resolved")
        # 更新后应该在 resolved 状态
        with db._lock:
            cursor = db._conn.execute("SELECT status FROM tickets WHERE id = 'TK-001'")
            row = cursor.fetchone()
            assert row[0] == "resolved"
    test("ticket_update", test_ticket_update)

    def test_open_tickets():
        open_tickets = db.get_open_tickets("u1")
        # TK-001 已解决，TK-002 仍 open
        assert len(open_tickets) == 1
        assert open_tickets[0]["id"] == "TK-002"
    test("open_tickets", test_open_tickets)

    def test_fts_search():
        results = db.search_messages("话费")
        assert len(results) >= 1
        assert any("话费" in r["content"] for r in results)
    test("fts_search", test_fts_search)

    def test_fts_search_chinese():
        # "话费" 在 messages 中有 "我想查话费"
        results = db.search_messages("查话费")
        assert len(results) >= 1
    test("fts_search_chinese", test_fts_search_chinese)

    db.close()

    # 测试 MemoryManager
    print("\n[MemoryManager]")
    mm = MemoryManager(db_path)

    def test_memory_create_session():
        mm.create_session("s3", user_id="u3")
    test("memory_create_session", test_memory_create_session)

    def test_memory_add_message():
        mm.add_message("s3", "user", "帮我查一下流量")
        mm.add_message("s3", "assistant", "好的，正在为您查询...")
    test("memory_add_message", test_memory_add_message)

    def test_memory_recall():
        context = mm.recall_user_context("u1")
        assert "张三" in context or len(context) > 0
    test("memory_recall", test_memory_recall)

    def test_memory_search():
        results = mm.search_history("流量")
        assert isinstance(results, list)
    test("memory_search", test_memory_search)

    def test_freeze_snapshot():
        mm.save_user_profile("u3", name="李四")
        mm.freeze_snapshot("u3")
        profile = mm.get_frozen_profile()
        assert profile is not None
        assert "李四" in profile
    test("freeze_snapshot", test_freeze_snapshot)

    mm.close()

finally:
    # 清理临时文件
    try:
        os.unlink(db_path)
    except:
        pass

print("\n" + "=" * 50)
print(f"结果: {passed} 通过, {failed} 失败")
print("=" * 50)

sys.exit(0 if failed == 0 else 1)
