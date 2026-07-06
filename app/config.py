import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"
INDEX_DIR = PAPERS_DIR / ".index"
SUBTOPICS_FILE = BASE_DIR / "subtopics.yaml"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

ARXIV_MIN_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL", "3.0"))
ARXIV_MAX_RETRIES = int(os.getenv("ARXIV_MAX_RETRIES", "5"))
ARXIV_BACKOFF_BASE = float(os.getenv("ARXIV_BACKOFF_BASE", "4.0"))
ARXIV_BACKOFF_CAP = float(os.getenv("ARXIV_BACKOFF_CAP", "60.0"))

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
