import os
from pathlib import Path

PHOENIX_HOME = Path(os.getenv("PHOENIX_HOME", Path.home() / ".phoenix"))
DATA_DIR = PHOENIX_HOME / "phoenix_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(exist_ok=True)

CONVERSATION_DB_PATH = DATA_DIR / "conversations.db"
SKILLS_DIR = DATA_DIR / "skills"
SKILLS_DIR.mkdir(exist_ok=True)
TRAINING_DATA_DIR = DATA_DIR / "training_data"
TRAINING_DATA_DIR.mkdir(exist_ok=True)

LLM_MODEL = os.getenv("PHOENIX_MODEL", "llama3.1:8b")
LLM_BASE_URL = os.getenv("PHOENIX_LLM_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("PHOENIX_EMBED_MODEL", "nomic-embed-text")

MAX_WORKING_MEMORY_TOKENS = int(os.getenv("PHOENIX_MAX_MEMORY", 2000))
