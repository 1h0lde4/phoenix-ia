import os
import uuid
import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from orchestrator import PhoenixOrchestrator
from config import CONVERSATION_DB_PATH

app = FastAPI(title="Phoenix AI", version="0.2.0")
orchestrator = PhoenixOrchestrator()

# ---------- Static files (portal) ----------
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def portal():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Phoenix Portal not found</h1>", status_code=404)

# ---------- API Models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "phoenix"
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    user_msg = request.messages[-1].content
    try:
        answer = orchestrator.process_input(user_msg, session_id="default")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=0,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage()
    )
    return response

@app.post("/v1/chat/completions/stream")
async def stream_chat(request: ChatCompletionRequest):
    return {"detail": "Streaming not implemented yet"}

@app.get("/api/status")
async def dev_status():
    skill_count = len(orchestrator.skill_registry.skills)
    skill_names = list(orchestrator.skill_registry.skills.keys())
    try:
        memory_count = orchestrator.semantic_memory.vectorstore._collection.count()
    except:
        memory_count = 0
    try:
        fts_count = sqlite3.connect(CONVERSATION_DB_PATH).execute("SELECT count(*) FROM messages_fts").fetchone()[0]
    except:
        fts_count = 0

    return {
        "model": os.getenv("PHOENIX_MODEL_PATH", "llama-3.2-1b-instruct.Q4_K_M.gguf"),
        "skills": {"count": skill_count, "names": skill_names},
        "memory": {"total_facts": memory_count, "fts_indexed": fts_count, "method": "RRF (semantic + keyword)"},
        "training": {"status": "idle", "last_run": "never"},
        "logs": [
            "Server started",
            f"Loaded {skill_count} skills",
            "Semantic memory online",
            f"FTS5 index ready ({fts_count} entries)"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
