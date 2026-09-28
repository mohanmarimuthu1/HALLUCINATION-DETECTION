"""
Configuration settings for the Hallucination Detection System
"""
import os
from dotenv import load_dotenv

# Paths below resolve against this file, not the working directory, so the
# app finds its vector store and data whether it is launched from the repo
# root (`streamlit run legacy/app.py`) or from inside legacy/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# .env lives at the repo root, shared with v2. load_dotenv() on its own
# searches upward from the caller, which would miss it from some CWDs.
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))

# ============================================
# API Configuration
# ============================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# OpenRouter API Configuration (DeepSeek)
# List of API keys for fallback (will try next key if current one fails)
OPENROUTER_API_KEYS = [
    key for key in (
        os.getenv("OPENROUTER_API_KEY_1", ""),
        os.getenv("OPENROUTER_API_KEY_2", ""),
    ) if key
]
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else ""  # Default to first key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"  # DeepSeek model via OpenRouter

# ============================================
# Model Configuration
# ============================================
# LLM Model (use the correct model name)
LLM_MODEL = "gemma-3-27b-it"
LLM_TEMPERATURE = 0.3  # Lower temperature for more factual responses

# Embedding Model (runs locally - free!)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ============================================
# RAG Configuration
# ============================================
# Text Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_DOCUMENTS = 3  # Number of documents to retrieve

# ============================================
# Vector Database Configuration
# ============================================
CHROMA_PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_base"

# ============================================
# Hallucination Detection Configuration
# ============================================
# Thresholds for hallucination scoring
HALLUCINATION_THRESHOLD_LOW = 0.3    # Below this is "Factual"
HALLUCINATION_THRESHOLD_HIGH = 0.7   # Above this is "Hallucinated"

# ============================================
# Paths
# ============================================
DATA_DIRECTORY = os.path.join(BASE_DIR, "data")
KNOWLEDGE_BASE_FILE = os.path.join(BASE_DIR, "data", "knowledge_base.txt")
