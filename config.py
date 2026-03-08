"""
Configuration settings for the Hallucination Detection System
"""
import os

# ============================================
# API Configuration
# ============================================
GOOGLE_API_KEY = "#######"

# OpenRouter API Configuration (DeepSeek)
# List of API keys for fallback (will try next key if current one fails)
OPENROUTER_API_KEYS = [
    "#####",
    "s#####",
    "sk#####",
    "sk-#####p",
    "sk-or-v1-a######a944ea831f5e1eb37788",  # Original key as fallback
    "sk-or-v1-######861a8c39d"
]
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0]  # Default to first key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"  # DeepSeek model via OpenRouter

# ============================================
# Model Configuration
# ============================================
# LLM Model (use the correct model name)
LLM_MODEL = "gemini-2.0-flash"
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
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
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
DATA_DIRECTORY = "./data"
KNOWLEDGE_BASE_FILE = "./data/knowledge_base.txt"
