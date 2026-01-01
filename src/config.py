import os
from dotenv import load_dotenv

load_dotenv()

# Index versioning for Qdrant collections
INDEX_VERSION = "v5_mmr_fix"

# Qdrant Connection Settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_documents"

# Model Configuration
# Embedding model: Multilingual MiniLM for efficient semantic search
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# LLM: GPT-OSS-20B via Groq (or compatible API)
LLM_MODEL = "openai/gpt-oss-20b"

# Chunking Strategy
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval Configuration ---
TOP_K_RETRIEVAL = 50  # Initial retrieval (before reranking)
TOP_K_RERANK = 7      # Final context chunks (after reranking)

# --- API Configuration ---
API_HOST = "0.0.0.0"
API_PORT = 8000

# --- MLOps Configuration ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate critical env vars
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment")

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    print("⚠️  WARNING: Langfuse credentials not set. Tracing disabled.")

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Feature Flags ---
ENABLE_QUERY_EXPANSION = True  # Multi-query retrieval
ENABLE_RERANKING = True        # FlashRank reranking
ENABLE_CACHING = True          # LangChain cache

# --- Performance Tuning ---
MAX_CONCURRENT_REQUESTS = 10   # For FastAPI
BACKGROUND_TASKS_LIMIT = 5     # Max simultaneous ingestions