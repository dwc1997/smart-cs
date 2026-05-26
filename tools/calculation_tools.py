"""计算工具

安全的数学计算和费用查询。
"""

import ast
import operator
import logging
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)

# 安全的运算符映射
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """安全的 AST 表达式求值。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _OPS[op_type](operand)
    else:
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


def calculator_tool(expression: str) -> Dict[str, Any]:
    """安全的数学计算器。

    支持：+, -, *, /, **, %, 以及括号。
    """
    try:
        expression = expression.strip()
        if not expression:
            return {"success": False, "error": "表达式不能为空"}

        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)

        # 格式化结果
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            result = int(result)

        return {
            "success": True,
            "expression": expression,
            "result": result,
        }
    except ZeroDivisionError:
        return {"success": False, "error": "除零错误"}
    except (ValueError, SyntaxError) as e:
        return {"success": False, "error": f"表达式错误: {e}"}
    except Exception as e:
        return {"success": False, "error": f"计算失败: {e}"}


# 模拟的资费数据
_TARIFF_DATA = {
    "5G畅享套餐": {"monthly_fee": 128, "data_gb": 30, "voice_min": 500, "overage_data": 5, "overage_voice": 0.15},
    "4G飞享套餐": {"monthly_fee": 58, "data_gb": 10, "voice_min": 200, "overage_data": 10, "overage_voice": 0.20},
    "大流量套餐": {"monthly_fee": 198, "data_gb": 100, "voice_min": 1000, "overage_data": 3, "overage_voice": 0.10},
}


def billing_query_tool(query_type: str, package_name: str = None, **kwargs) -> Dict[str, Any]:
    """费用查询工具。

    query_type: "package_info" | "estimate_bill" | "compare_packages"
    """
    try:
        if query_type == "package_info":
            if not package_name:
                return {
                    "success": True,
                    "available_packages": list(_TARIFF_DATA.keys()),
                    "message": "请指定套餐名称以查看详情",
                }
            info = _TARIFF_DATA.get(package_name)
            if not info:
                return {"success": False, "error": f"未找到套餐: {package_name}"}
            return {
                "success": True,
                "package": package_name,
                "info": info,
            }

        elif query_type == "estimate_bill":
            # 估算账单
            if not package_name:
                return {"success": False, "error": "需要指定套餐名称"}
            info = _TARIFF_DATA.get(package_name)
            if not info:
                return {"success": False, "error": f"未找到套餐: {package_name}"}

            data_used = kwargs.get("data_used_gb", 0)
            voice_used = kwargs.get("voice_used_min", 0)

            total = info["monthly_fee"]
            data_overage = max(0, data_used - info["data_gb"])
            voice_overage = max(0, voice_used - info["voice_min"])

            if data_overage > 0:
                total += data_overage * info["overage_data"]
            if voice_overage > 0:
                total += voice_overage * info["overage_voice"]

            return {
                "success": True,
                "package": package_name,
                "base_fee": info["monthly_fee"],
                "data_used_gb": data_used,
                "data_included_gb": info["data_gb"],
                "data_overage_gb": data_overage,
                "data_overage_fee": data_overage * info["overage_data"],
                "voice_used_min": voice_used,
                "voice_included_min": info["voice_min"],
                "voice_overage_min": voice_overage,
                "voice_overage_fee": voice_overage * info["overage_voice"],
                "estimated_total": total,
            }

        elif query_type == "compare_packages":
            return {
                "success": True,
                "packages": _TARIFF_DATA,
            }

        else:
            return {"success": False, "error": f"未知查询类型: {query_type}"}

    except Exception as e:
        logger.error(f"Billing query failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# 注册工具
# ============================================================

CALCULATOR_SCHEMA = {
    "name": "calculator",
    "description": "安全的数学计算器，支持加减乘除、幂运算、取模和括号",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '(128 + 50) * 1.05'",
            },
        },
        "required": ["expression"],
    },
}

BILLING_QUERY_SCHEMA = {
    "name": "billing_query",
    "description": "查询套餐资费、估算账单、对比套餐",
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "description": "查询类型",
                "enum": ["package_info", "estimate_bill", "compare_packages"],
            },
            "package_name": {
                "type": "string",
                "description": "套餐名称（package_info 和 estimate_bill 时需要）",
            },
            "data_used_gb": {
                "type": "number",
                "description": "已使用流量 GB（estimate_bill 时使用）",
            },
            "voice_used_min": {
                "type": "integer",
                "description": "已使用通话分钟（estimate_bill 时使用）",
            },
        },
        "required": ["query_type"],
    },
}

registry.register(
    name="calculator",
    toolset="calculation",
    schema=CALCULATOR_SCHEMA,
    handler=lambda args, **kw: calculator_tool(expression=args.get("expression", "")),
    description="安全数学计算器",
)

registry.register(
    name="billing_query",
    toolset="calculation",
    schema=BILLING_QUERY_SCHEMA,
    handler=lambda args, **kw: billing_query_tool(
        query_type=args.get("query_type", "compare_packages"),
        package_name=args.get("package_name"),
        data_used_gb=args.get("data_used_gb", 0),
        voice_used_min=args.get("voice_used_min", 0),
    ),
    description="费用查询",
)
