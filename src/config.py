import os
from dotenv import load_dotenv

load_dotenv()

# Index versioning for Qdrant collections
INDEX_VERSION = "v5_mmr_fix"

# Qdrant Connection Settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = f"documents_{INDEX_VERSION}"

# Model Configuration
# Embedding model: Multilingual MiniLM for efficient semantic search
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# LLM: GPT-OSS-20B via Groq (or compatible API)
LLM_MODEL = "openai/gpt-oss-20b"

# Chunking Strategy
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200