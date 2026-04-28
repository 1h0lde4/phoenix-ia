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

        # Multi-signal recall
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
            del self.sessions[session_id]import json
import re
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from memory import create_working_memory, SemanticMemoryStore
from skill_registry import SkillRegistry
from config import LLM_MODEL, LLM_BASE_URL

class PhoenixOrchestrator:
    def __init__(self):
        self.llm = ChatOllama(model=LLM_MODEL, base_url=LLM_BASE_URL, temperature=0.7)
        self.skill_registry = SkillRegistry()
        self.semantic_memory = SemanticMemoryStore()
        self.sessions = {}

    def get_working_memory(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = create_working_memory(self.llm, session_id)
        return self.sessions[session_id]

    def _build_system_prompt(self, tools_schema):
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in tools_schema]
        )
        prompt = f"""You are Phoenix, a self-evolving AI assistant. You have access to tools and long-term memory.

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
        history_msgs = wm.chat_memory.messages
        user_msg = HumanMessage(content=f"Relevant memories:\n{memory_context}\n\nUser: {user_input}")

        messages = [system_msg] + history_msgs + [user_msg]
        response = self.llm(messages)
        assistant_text = response.content.strip()

        tool_name, tool_params = self._parse_tool_call(assistant_text)
        if tool_name:
            try:
                tool_result = self.skill_registry.execute(tool_name, **tool_params)
            except Exception as e:
                tool_result = f"Tool error: {str(e)}"
            # update working memory with the whole exchange
            wm.chat_memory.add_message(HumanMessage(content=user_input))
            wm.chat_memory.add_message(AIMessage(content=assistant_text))
            wm.chat_memory.add_message(AIMessage(content=f"Tool result: {tool_result}"))
            final_messages = [system_msg] + wm.chat_memory.messages
            final_response = self.llm(final_messages)
            final_text = final_response.content.strip()
            wm.chat_memory.add_message(AIMessage(content=final_text))
            self.semantic_memory.add_memory(f"User asked: {user_input}\nAssistant answered: {final_text}")
            return final_text
        else:
            wm.chat_memory.add_message(HumanMessage(content=user_input))
            wm.chat_memory.add_message(AIMessage(content=assistant_text))
            self.semantic_memory.add_memory(f"User asked: {user_input}\nAssistant answered: {assistant_text}")
            return assistant_text

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
