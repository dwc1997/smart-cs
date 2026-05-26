import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM 配置
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://maas-apigateway.dt.zte.com.cn/model/qwq-32b/v1",
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwq-32b")

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048

# ============================================================
# LangSmith 配置（可选，通过环境变量启用）
# ============================================================
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "smart-cs")

# ============================================================
# 客服 Agent 配置
# ============================================================
MAX_ITERATIONS = 15
INTENT_CONFIRM_THRESHOLD = 0.7
CONTEXT_LENGTH = 32768
CONTEXT_COMPRESS_THRESHOLD = 0.50

# ============================================================
# 数据库路径
# ============================================================
SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH", "data/sessions.db")
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", "data/knowledge.db")

# ============================================================
# Prompt 文件路径
# ============================================================
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
