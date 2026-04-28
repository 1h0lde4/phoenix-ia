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

# LLM settings – now using a local GGUF file
LLM_MODEL_PATH = os.getenv(
    "PHOENIX_MODEL_PATH",
    str(PHOENIX_HOME / "models" / "llama-3.2-1b-instruct.Q4_K_M.gguf")
)
EMBEDDING_MODEL = os.getenv("PHOENIX_EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_WORKING_MEMORY_TOKENS = int(os.getenv("PHOENIX_MAX_MEMORY", 2000))

# Memory priority directories
P1_RULES_DIR = DATA_DIR / "memory" / "rules"
P1_RULES_DIR.mkdir(parents=True, exist_ok=True)
