import json
import re
import datetime
from pathlib import Path
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memory import WorkingMemory, SemanticMemoryStore, setup_fts
from skill_registry import SkillRegistry
from config import LLM_MODEL_PATH, P1_RULES_DIR, AUTO_IMPROVE_INTERVAL, AGENTS_DIR

class PhoenixOrchestrator:
    def __init__(self):
        self.llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.7,
    max_tokens=512,                # can now be generous
    n_ctx=8192,                    # Mistral's native 8K context
    chat_format="mistral-instruct", # use Mistral's chat template
    verbose=False
)
        self.skill_registry = SkillRegistry()
        self.semantic_memory = SemanticMemoryStore()
        self.sessions = {}
        self.message_count = 0
        setup_fts()
        self._load_personas()

    def _load_personas(self):
        self.personas = {}
        if AGENTS_DIR.exists():
            for file in AGENTS_DIR.glob("*.md"):
                name = file.stem
                self.personas[name] = file.read_text()

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

    def _trim_memory(self, text, max_chars=200):
        """Shorten memory text to max_chars to save tokens."""
        return text[:max_chars] + "..." if len(text) > max_chars else text

    def process_input(self, user_input: str, session_id="default"):
        wm = self.get_working_memory(session_id)

        # Retrieve and shorten memories
        relevant_memories = self.semantic_memory.recall(user_input, k=2)  # k=2 to save space
        memory_context = "\n".join([f"- {self._trim_memory(m)}" for m in relevant_memories])

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
        """Autonomous self‑improvement step: find gaps, search web, ingest."""
        gap_prompt = """You are Phoenix's self-analysis module.
Review the last few conversations (provided below as memory) and list any topics you were unsure about or lacked current information.
Output a JSON list of strings, e.g. ["topic1", "topic2"]. If none, output [].
Conversations:
"""
        recent_memories = self.semantic_memory.recall("recent conversation", k=5)
        gap_prompt += "\n".join(recent_memories[:2000])  # limit chars
        gap_response = self.llm.invoke(gap_prompt)
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
            trainer_response = self.llm.invoke(trainer_prompt)
            self.semantic_memory.add_improvement_log(
                topic="training_decision",
                summary=trainer_response.strip()[:400]
            )

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
