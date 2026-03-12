import os
from dotenv import load_dotenv

load_dotenv()

# LLM Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LOCAL_MODEL = os.getenv("LOCAL_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

# Data Sources
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Storage
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")

# RAG Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 75))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.75))