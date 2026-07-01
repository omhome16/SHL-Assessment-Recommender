"""
Configuration module for the SHL Assessment Recommender.
All settings are driven by environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "SHL_product_catlog.json"

# ── LLM Settings (Groq) ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Retrieval Settings ────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "25"))

# ── Conversation Settings ─────────────────────────────────────────────────
MAX_TURNS = 8           # Hard cap: 8 turns total (user + assistant combined)
RESPONSE_TIMEOUT = 28   # Seconds; leaves 2s buffer under the 30s limit
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1
