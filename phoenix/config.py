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

LLM_MODEL_PATH = os.getenv(
    "PHOENIX_MODEL_PATH",
    str(PHOENIX_HOME / "models" / "llama-3.2-1b-instruct.Q4_K_M.gguf")
)
EMBEDDING_MODEL = os.getenv("PHOENIX_EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_WORKING_MEMORY_TOKENS = int(os.getenv("PHOENIX_MAX_MEMORY", 2000))

P1_RULES_DIR = DATA_DIR / "memory" / "rules"
P1_RULES_DIR.mkdir(parents=True, exist_ok=True)

# NEW: auto‑improvement trigger
AUTO_IMPROVE_INTERVAL = int(os.getenv("PHOENIX_AUTO_IMPROVE_INTERVAL", "5"))  # every 5 messages

# NEW: Personas directory
AGENTS_DIR = DATA_DIR / "agents"
AGENTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY_PATH = DATA_DIR / "model_registry.json"

# Default models (these will be auto-created if registry is empty)
DEFAULT_MODELS = [
    {
        "name": "mistral-7b",
        "path": str(PHOENIX_HOME / "models" / "mistral-7b-instruct-v0.3.Q4_K_M.gguf"),
        "tags": ["reasoning", "general"],
        "n_ctx": 8192,
        "chat_format": "mistral-instruct"
    },
    {
        "name": "tinyllama-1.1b",
        "path": str(PHOENIX_HOME / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        "tags": ["simple", "fast"],
        "n_ctx": 2048,
        "chat_format": "chatml"
    },
    # Codestral is optional; download only if you want
    # {
    #     "name": "codestral-22b",
    #     "path": str(PHOENIX_HOME / "models" / "codestral-22b.Q4_K_M.gguf"),
    #     "tags": ["code"],
    #     "n_ctx": 8192,
    #     "chat_format": "mistral-instruct"   # Codestral uses Mistral format
    # }
]
