"""Central configuration. Everything comes from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

ORACLE_USER = os.getenv("ORACLE_USER", "vector_user")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "VectorPass_123")
ORACLE_DSN = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1")

EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, runs locally (no API quota used)
EMBED_DIM = 384

MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))
GUARDRAIL_LLM = os.getenv("GUARDRAIL_LLM", "1") == "1"
GUARDRAIL_FAIL_CLOSED = os.getenv("GUARDRAIL_FAIL_CLOSED", "1") == "1"
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
EVAL_DELAY_SECONDS = float(os.getenv("EVAL_DELAY_SECONDS", "8"))

LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data" / "pdfs"


def require_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "No Gemini key found. Put GEMINI_API_KEY=... (or GOOGLE_API_KEY) in .env"
        )
    return GEMINI_API_KEY
