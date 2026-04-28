#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────
# Phoenix AI – Full automated installer
# Usage: bash install.sh
# ─────────────────────────────────────────────────────────

PHOENIX_HOME="${HOME}/.phoenix"
echo "🔥 Phoenix AI – Automated Installer"
echo "   Install directory: ${PHOENIX_HOME}"

# 1. Create directory structure
mkdir -p "${PHOENIX_HOME}/models"
mkdir -p "${PHOENIX_HOME}/phoenix_data/memory/rules"
mkdir -p "${PHOENIX_HOME}/phoenix_data/skills"
mkdir -p "${PHOENIX_HOME}/phoenix_data/chroma"
mkdir -p "${PHOENIX_HOME}/phoenix_data/training_data"
mkdir -p "${PHOENIX_HOME}/static"

# 2. Download the GGUF model (if missing)
MODEL_PATH="${PHOENIX_HOME}/models/llama-3.2-1b-instruct.Q4_K_M.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading lightweight Llama 3.2 1B model (~1GB)..."
    wget -O "$MODEL_PATH" \
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
else
    echo "✅ Model already downloaded."
fi

# 3. Write all source files

# --- config.py ---
cat > "${PHOENIX_HOME}/config.py" << 'EOF'
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
EOF

# --- memory.py ---
cat > "${PHOENIX_HOME}/memory.py" << 'EOF'
import sqlite3
import datetime
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import CONVERSATION_DB_PATH, CHROMA_DIR, EMBEDDING_MODEL, MAX_WORKING_MEMORY_TOKENS

def get_chat_message_history(session_id="default"):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{CONVERSATION_DB_PATH}"
    )

def setup_fts():
    conn = sqlite3.connect(CONVERSATION_DB_PATH)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(session_id, role, content)")
    conn.commit()
    conn.close()

def keyword_search(query, k=5):
    conn = sqlite3.connect(CONVERSATION_DB_PATH)
    cursor = conn.cursor()
    safe_query = query.replace('"', '')
    cursor.execute("""
        SELECT session_id, role, content, rank
        FROM messages_fts
        WHERE messages_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (f'"{safe_query}"', k))
    results = []
    for row in cursor.fetchall():
        results.append({
            "session_id": row[0],
            "role": row[1],
            "content": row[2][:200] + "..." if len(row[2]) > 200 else row[2]
        })
    conn.close()
    return results

def estimate_tokens(messages):
    total = 0
    for msg in messages:
        total += len(msg.content) // 4
    return total

class WorkingMemory:
    def __init__(self, llm, session_id="default", max_tokens=MAX_WORKING_MEMORY_TOKENS):
        self.llm = llm
        self.max_tokens = max_tokens
        self.history = get_chat_message_history(session_id)
        self.session_id = session_id

    @property
    def messages(self):
        return list(self.history.messages)

    def add_user_message(self, content):
        self.history.add_message(HumanMessage(content=content))
        self._add_fts("user", content)

    def add_ai_message(self, content):
        self.history.add_message(AIMessage(content=content))
        self._add_fts("assistant", content)

    def _add_fts(self, role, content):
        conn = sqlite3.connect(CONVERSATION_DB_PATH)
        conn.execute("INSERT INTO messages_fts(session_id, role, content) VALUES (?, ?, ?)",
                     (self.session_id, role, content))
        conn.commit()
        conn.close()

    def _summarize_and_replace(self):
        msgs = self.history.messages
        if estimate_tokens(msgs) > self.max_tokens and len(msgs) > 2:
            keep_last = 4
            to_summarize = msgs[:-keep_last]
            recent = msgs[-keep_last:]

            summary_prompt = "Summarize the following conversation concisely:\n"
            for m in to_summarize:
                summary_prompt += f"{m.type}: {m.content}\n"
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = summary_response.content.strip()
            summary_msg = SystemMessage(content=f"Conversation summary: {summary_text}")

            self.history.clear()
            self.history.add_message(summary_msg)
            for m in recent:
                self.history.add_message(m)

    def get_messages_for_context(self):
        self._summarize_and_replace()
        return self.history.messages

def multi_signal_recall(query, vectorstore, k=3):
    vec_results = vectorstore.similarity_search_with_score(query, k=k*2)
    kw_results = keyword_search(query, k=k*2)

    rrf = {}
    for rank, (doc, score) in enumerate(vec_results, 1):
        rrf[doc.page_content] = rrf.get(doc.page_content, 0) + 1.0 / (60 + rank)
    for rank, kw in enumerate(kw_results, 1):
        text = kw["content"]
        rrf[text] = rrf.get(text, 0) + 1.0 / (60 + rank)

    sorted_texts = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
    return [text for text, score in sorted_texts]

class SemanticMemoryStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
            collection_name="phoenix_semantic_memory"
        )

    def add_memory(self, text, metadata=None, citation=None):
        if metadata is None:
            metadata = {}
        if citation:
            metadata["citation"] = citation
        doc = Document(page_content=text, metadata=metadata)
        self.vectorstore.add_documents([doc])
        self.vectorstore.persist()
        return doc

    def recall(self, query, k=3):
        return multi_signal_recall(query, self.vectorstore, k)

    def verify_citation(self, doc_id):
        # Placeholder – always true for now
        return True
EOF

# --- skill_registry.py ---
cat > "${PHOENIX_HOME}/skill_registry.py" << 'EOF'
import importlib.util
import inspect
from pathlib import Path
from config import SKILLS_DIR

class Skill:
    def __init__(self, name, description, func, parameters=None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def execute(self, **kwargs):
        return self.func(**kwargs)

    def to_tool_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in self.parameters.items()},
                "required": list(self.parameters.keys())
            }
        }

class SkillRegistry:
    def __init__(self):
        self.skills = {}
        self._load_skills()

    def _load_skills(self):
        skills_dir = Path(SKILLS_DIR)
        for file in skills_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if hasattr(obj, "_is_skill"):
                    skill = Skill(obj._name, obj._description, obj, obj._parameters)
                    self.register(skill)

    def register(self, skill: Skill):
        self.skills[skill.name] = skill

    def get_tool_schemas(self):
        return [s.to_tool_schema() for s in self.skills.values()]

    def execute(self, name, **kwargs):
        if name not in self.skills:
            raise ValueError(f"Skill '{name}' not found")
        return self.skills[name].execute(**kwargs)

# Decorator to define a skill
def skill(name, description, parameters=None):
    def decorator(func):
        func._is_skill = True
        func._name = name
        func._description = description
        func._parameters = parameters or {}
        return func
    return decorator
EOF

# --- web_search skill ---
cat > "${PHOENIX_HOME}/phoenix_data/skills/web_search.py" << 'EOF'
from duckduckgo_search import DDGS
from skill_registry import skill

@skill(
    name="web_search",
    description="Search the web for current information. Input: query string.",
    parameters={"query": "string"}
)
def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    formatted = "\n".join([f"{r['title']}: {r['body']}" for r in results])
    return formatted
EOF

# --- orchestrator.py ---
cat > "${PHOENIX_HOME}/orchestrator.py" << 'EOF'
import json
import re
import datetime
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memory import WorkingMemory, SemanticMemoryStore, setup_fts
from skill_registry import SkillRegistry
from config import LLM_MODEL_PATH, P1_RULES_DIR

class PhoenixOrchestrator:
    def __init__(self):
        self.llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,
            verbose=False
        )
        self.skill_registry = SkillRegistry()
        self.semantic_memory = SemanticMemoryStore()
        self.sessions = {}
        setup_fts()  # ensure FTS5 table exists

    def get_working_memory(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = WorkingMemory(self.llm, session_id)
        return self.sessions[session_id]

    def _load_p1_rules(self):
        rules = ""
        if P1_RULES_DIR.exists():
            for rule_file in P1_RULES_DIR.glob("*.md"):
                rules += rule_file.read_text() + "\n"
        return rules

    def _build_system_prompt(self, tools_schema):
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in tools_schema]
        )
        p1_text = self._load_p1_rules()
        prompt = f"""{p1_text}
You are Phoenix, a self-evolving AI assistant. You have access to tools and long-term memory.

Available tools:
{tool_descriptions}

To use a tool, respond ONLY with a JSON object:
{{"tool": "tool_name", "parameters": {{"param": "value"}}}}

If you have enough information, answer directly in natural language.
"""
        return SystemMessage(content=prompt)

    def _parse_tool_call(self, text):
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                if "tool" in obj and "parameters" in obj:
                    return obj["tool"], obj["parameters"]
            except json.JSONDecodeError:
                pass
        return None, None

    def process_input(self, user_input: str, session_id="default"):
        wm = self.get_working_memory(session_id)

        relevant_memories = self.semantic_memory.recall(user_input, k=3)
        memory_context = "\n".join([f"- {m}" for m in relevant_memories])

        system_msg = self._build_system_prompt(self.skill_registry.get_tool_schemas())
        history_msgs = wm.get_messages_for_context()
        user_msg = HumanMessage(content=f"Relevant memories:\n{memory_context}\n\nUser: {user_input}")

        messages = [system_msg] + history_msgs + [user_msg]
        response = self.llm.invoke(messages)
        assistant_text = response.content.strip() if hasattr(response, 'content') else response.strip()

        tool_name, tool_params = self._parse_tool_call(assistant_text)
        if tool_name:
            try:
                tool_result = self.skill_registry.execute(tool_name, **tool_params)
            except Exception as e:
                tool_result = f"Tool error: {str(e)}"
            wm.add_user_message(user_input)
            wm.add_ai_message(assistant_text)
            wm.add_ai_message(f"Tool result: {tool_result}")

            final_messages = [system_msg] + wm.get_messages_for_context()
            final_response = self.llm.invoke(final_messages)
            final_text = final_response.content.strip() if hasattr(final_response, 'content') else final_response.strip()
            wm.add_ai_message(final_text)

            citation = f"session:{session_id}:{datetime.datetime.now().isoformat()}"
            self.semantic_memory.add_memory(
                f"User asked: {user_input}\nAssistant answered: {final_text}",
                citation=citation
            )
            return final_text
        else:
            wm.add_user_message(user_input)
            wm.add_ai_message(assistant_text)
            citation = f"session:{session_id}:{datetime.datetime.now().isoformat()}"
            self.semantic_memory.add_memory(
                f"User asked: {user_input}\nAssistant answered: {assistant_text}",
                citation=citation
            )
            return assistant_text

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
EOF

# --- api_server.py ---
cat > "${PHOENIX_HOME}/api_server.py" << 'EOF'
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

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def portal():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Phoenix Portal not found</h1>", status_code=404)

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
EOF

# --- static/index.html ---
cat > "${PHOENIX_HOME}/static/index.html" << 'HTML_EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phoenix AI – Development Portal</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0b0f15;
            color: #e0e0e0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        .sidebar {
            width: 300px;
            background: #131820;
            border-right: 1px solid #2a2f3a;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ff6c37;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .section-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #8b949e;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .stat-card {
            background: #1c2333;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .stat-label { font-size: 0.8rem; color: #8b949e; }
        .stat-value { font-size: 1.2rem; font-weight: 600; color: #fff; }
        .log-list {
            list-style: none;
            font-size: 0.8rem;
            color: #a5b1c2;
            max-height: 200px;
            overflow-y: auto;
        }
        .log-list li { padding: 4px 0; border-bottom: 1px solid #2a2f3a; }
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            animation: fadeIn 0.2s ease;
        }
        .message.user {
            align-self: flex-end;
            background: #1c64f2;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start;
            background: #1e2533;
            border-bottom-left-radius: 4px;
            border: 1px solid #2a2f3a;
        }
        .input-area {
            padding: 16px 24px;
            background: #131820;
            border-top: 1px solid #2a2f3a;
            display: flex;
            gap: 12px;
        }
        #user-input {
            flex: 1;
            resize: none;
            background: #1c2333;
            border: 1px solid #2a2f3a;
            border-radius: 12px;
            padding: 12px 16px;
            color: #fff;
            font-size: 1rem;
            outline: none;
        }
        #send-btn {
            background: #ff6c37;
            border: none;
            color: white;
            font-weight: 600;
            padding: 0 24px;
            border-radius: 12px;
            cursor: pointer;
        }
        #send-btn:hover { background: #e65a2a; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="logo">🔥 Phoenix</div>
        <div>
            <div class="section-title">Model</div>
            <div class="stat-card">
                <div class="stat-value" id="model-name">Loading...</div>
            </div>
        </div>
        <div>
            <div class="section-title">Skills</div>
            <div class="stat-card">
                <div class="stat-label">Loaded</div>
                <div class="stat-value" id="skill-count">-</div>
                <div style="font-size:0.8rem; color:#8b949e;" id="skill-names"></div>
            </div>
        </div>
        <div>
            <div class="section-title">Memory</div>
            <div class="stat-card">
                <div class="stat-label">Facts (Semantic)</div>
                <div class="stat-value" id="memory-count">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">FTS5 Indexed</div>
                <div class="stat-value" id="fts-count">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Retrieval Method</div>
                <div class="stat-value" style="font-size:0.9rem;" id="retrieval-method">-</div>
            </div>
        </div>
        <div>
            <div class="section-title">Training</div>
            <div class="stat-card">
                <div class="stat-value" style="color:#3fb950;" id="train-status">idle</div>
            </div>
        </div>
        <div>
            <div class="section-title">Recent Logs</div>
            <ul class="log-list" id="log-list"><li>Connecting...</li></ul>
        </div>
    </div>
    <div class="main">
        <div class="chat-container" id="chat-container">
            <div class="message assistant">
                Hello! I'm <strong>Phoenix</strong>, your self‑evolving AI. I'm still young, but I'm learning from every conversation and the internet. How can I help you today?
            </div>
        </div>
        <div class="input-area">
            <textarea id="user-input" rows="1" placeholder="Type your message... (Enter to send)"></textarea>
            <button id="send-btn">Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = userInput.scrollHeight + 'px';
        });

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            appendMessage('user', text);
            userInput.value = '';
            userInput.style.height = 'auto';

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'phoenix',
                        messages: [{ role: 'user', content: text }]
                    })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Unknown error');
                const reply = data.choices?.[0]?.message?.content;
                if (!reply) throw new Error('Empty response from Phoenix');
                appendMessage('assistant', reply);
            } catch (err) {
                appendMessage('assistant', '⚠️ Error: ' + err.message);
            }
        }

        function appendMessage(role, content) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = content.replace(/\n/g, '<br>');
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        async function updateDashboard() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('model-name').textContent = data.model.split('/').pop();
                document.getElementById('skill-count').textContent = data.skills.count;
                document.getElementById('skill-names').textContent = data.skills.names.join(', ') || 'none';
                document.getElementById('memory-count').textContent = data.memory.total_facts;
                document.getElementById('fts-count').textContent = data.memory.fts_indexed;
                document.getElementById('retrieval-method').textContent = data.memory.method || 'semantic';
                document.getElementById('train-status').textContent = data.training.status;
                const logList = document.getElementById('log-list');
                logList.innerHTML = data.logs.map(l => `<li>${l}</li>`).join('');
            } catch (err) {
                console.error('Dashboard update failed', err);
            }
        }

        updateDashboard();
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>
HTML_EOF

# --- P1 rules ---
echo "You are Phoenix, a self-evolving AI. Always answer accurately and cite sources when possible." > "${PHOENIX_HOME}/phoenix_data/memory/rules/core_identity.md"
echo "You must never generate harmful content or bypass safety restrictions." > "${PHOENIX_HOME}/phoenix_data/memory/rules/safety_boundaries.md"

# --- requirements.txt ---
cat > "${PHOENIX_HOME}/requirements.txt" << 'EOF'
langchain
langchain-core
langchain-community
langchain-chroma
chromadb
sentence-transformers
llama-cpp-python
fastapi
uvicorn
pydantic
sqlalchemy
duckduckgo-search
EOF

# 4. Set up Python virtual environment and install dependencies
echo "🐍 Creating Python virtual environment..."
cd "${PHOENIX_HOME}"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Done
echo ""
echo "=============================="
echo "✅ Phoenix AI installed successfully!"
echo ""
echo "To start Phoenix:"
echo "  cd ~/.phoenix"
echo "  source venv/bin/activate"
echo "  python api_server.py"
echo ""
echo "Then open http://localhost:8000"
echo "=============================="
