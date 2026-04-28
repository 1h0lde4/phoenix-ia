import json
import re
import datetime
import threading
import time
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memory import WorkingMemory, SemanticMemoryStore, setup_fts
from skill_registry import SkillRegistry
from model_router import ModelRouter, ModelInfo
from config import (
    MODEL_REGISTRY_PATH, DEFAULT_MODELS,
    P1_RULES_DIR, AUTO_IMPROVE_INTERVAL, AGENTS_DIR,
    MEMORY_WORKER_INTERVAL, MEMORY_WORKER_ENABLED
)

logger = logging.getLogger("phoenix.orchestrator")
logging.basicConfig(level=logging.INFO)

class PhoenixOrchestrator:
    def __init__(self):
        self.router = ModelRouter(registry_path=MODEL_REGISTRY_PATH)
        if not self.router.models:
            for model_def in DEFAULT_MODELS:
                info = ModelInfo(**model_def)
                self.router.add_model(info)
        self.skill_registry = SkillRegistry()
        self.semantic_memory = SemanticMemoryStore()
        self.sessions = {}
        self.message_count = 0
        setup_fts()
        self._load_personas()
        if MEMORY_WORKER_ENABLED:
            self._start_memory_worker()

    def _load_personas(self):
        self.personas = {}
        if AGENTS_DIR.exists():
            for file in AGENTS_DIR.glob("*.md"):
                name = file.stem
                self.personas[name] = file.read_text()

    def get_working_memory(self, session_id):
        if session_id not in self.sessions:
            # Temporarily give it the reasoning model; will be overridden per request
            self.sessions[session_id] = WorkingMemory(self.router.get_model("reasoning"), session_id)
        return self.sessions[session_id]

    def _load_p1_rules(self):
        rules = ""
        if P1_RULES_DIR.exists():
            for rule_file in P1_RULES_DIR.glob("*.md"):
                rules += rule_file.read_text() + "\n"
        return rules

    def _detect_task_hint(self, user_input: str) -> str:
        code_keywords = ["write a function", "implement", "code", "debug", "program", "python", "javascript", "function", "class", "script"]
        simple_keywords = ["what is", "tell me a joke", "quick", "simple", "yes or no", "short answer"]
        if any(word in user_input.lower() for word in code_keywords):
            return "code"
        if any(word in user_input.lower() for word in simple_keywords):
            return "simple"
        return "reasoning"

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

    def _trim_memory(self, text, max_chars=200):
        return text[:max_chars] + "..." if len(text) > max_chars else text

    def process_input(self, user_input: str, session_id="default"):
        task_hint = self._detect_task_hint(user_input)
        llm = self.router.get_model(task_hint)

        wm = self.get_working_memory(session_id)
        wm.llm = llm   # ensure the working memory uses the same model for summarization

        relevant_memories = self.semantic_memory.recall(user_input, k=2)
        memory_context = "\n".join([f"- {self._trim_memory(m)}" for m in relevant_memories])

        system_msg = self._build_system_prompt(self.skill_registry.get_tool_schemas())
        history_msgs = wm.get_messages_for_context()
        user_msg = HumanMessage(content=f"Relevant memories:\n{memory_context}\n\nUser: {user_input}")

        messages = [system_msg] + history_msgs + [user_msg]
        response = llm.invoke(messages)
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
            final_response = llm.invoke(final_messages)
            final_text = final_response.content.strip() if hasattr(final_response, 'content') else final_response.strip()
            wm.add_ai_message(final_text)

            citation = f"session:{session_id}:{datetime.datetime.now().isoformat()}"
            self.semantic_memory.add_memory(
                f"User asked: {user_input}\nAssistant answered: {final_text}",
                citation=citation
            )
            self._count_message()
            return final_text
        else:
            wm.add_user_message(user_input)
            wm.add_ai_message(assistant_text)
            citation = f"session:{session_id}:{datetime.datetime.now().isoformat()}"
            self.semantic_memory.add_memory(
                f"User asked: {user_input}\nAssistant answered: {assistant_text}",
                citation=citation
            )
            self._count_message()
            return assistant_text

    def _count_message(self):
        self.message_count += 1
        if self.message_count % AUTO_IMPROVE_INTERVAL == 0:
            self.maybe_self_improve()

    def maybe_self_improve(self):
        llm = self.router.get_model("reasoning")
        gap_prompt = """You are Phoenix's self-analysis module.
Review the last few conversations and list any topics you were unsure about or lacked current information.
Output a JSON list of strings, e.g. ["topic1", "topic2"]. If none, output [].
Conversations:
"""
        recent_memories = self.semantic_memory.recall("recent conversation", k=5)
        gap_prompt += "\n".join(recent_memories[:2000])
        gap_response = llm.invoke(gap_prompt)
        gap_text = gap_response.content.strip() if hasattr(gap_response, 'content') else gap_response.strip()
        try:
            gaps = json.loads(gap_text)
            if not isinstance(gaps, list):
                gaps = []
        except:
            gaps = []

        if not gaps:
            return

        for topic in gaps[:2]:
            search_result = self.skill_registry.execute("web_search", query=topic)
            if search_result and "No results found." not in search_result:
                self.semantic_memory.add_improvement_log(
                    topic=topic,
                    summary=search_result[:400],
                    source_url="web_search"
                )

        if "phoenix-trainer" in self.personas:
            trainer_prompt = self.personas["phoenix-trainer"] + f"\n\nRecent gaps: {gaps}\nShould we schedule a training round?"
            trainer_response = llm.invoke(trainer_prompt)
            self.semantic_memory.add_improvement_log(
                topic="training_decision",
                summary=trainer_response.strip()[:400]
            )

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    # ---------- Proactive Memory Worker ----------
    def _start_memory_worker(self):
        def worker():
            while True:
                time.sleep(MEMORY_WORKER_INTERVAL)
                try:
                    self._run_memory_maintenance()
                except Exception as e:
                    logging.error(f"Memory worker error: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        logger.info("Proactive memory worker started.")

    def _run_memory_maintenance(self):
        logger.info("Memory worker: starting maintenance cycle")

        # 1. Citation verification (stub)
        recent_mems = self.semantic_memory.recall("citation:", k=10)
        stale_count = 0
        for mem_text in recent_mems:
            if "citation:" in mem_text and "stale" not in mem_text:
                stale_count += 1
        if stale_count > 0:
            self.semantic_memory.add_memory(
                f"Memory worker noticed {stale_count} unverified citations",
                metadata={"type": "worker_log"}
            )

        # 2. Duplicate detection (placeholder)
        duplicates = self.semantic_memory.find_duplicates()
        for dup_id in duplicates:
            self.semantic_memory.delete_memory_by_id(dup_id)
            self.semantic_memory.add_memory(f"Removed duplicate memory {dup_id}",
                                            metadata={"type": "worker_log"})

        # 3. Consolidation
        all_mems = self.semantic_memory.get_all_memories(limit=5)
        if all_mems and len(all_mems) >= 3:
            llm = self.router.get_model("reasoning")
            summary_prompt = "Summarize these recent Phoenix memories into a concise knowledge item:\n"
            for mem in all_mems:
                summary_prompt += f"- {mem[:200]}\n"
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = summary_response.content.strip()
            self.semantic_memory.add_memory(
                f"Consolidated knowledge: {summary_text}",
                metadata={"type": "consolidation"}
            )

        logger.info("Memory worker: maintenance cycle complete")
