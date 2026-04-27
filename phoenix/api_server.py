from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from orchestrator import PhoenixOrchestrator

app = FastAPI(title="Phoenix AI", version="0.1.0")
orchestrator = PhoenixOrchestrator()

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

# Optional: streaming stub
@app.post("/v1/chat/completions/stream")
async def stream_chat(request: ChatCompletionRequest):
    return {"detail": "Streaming not implemented yet"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
