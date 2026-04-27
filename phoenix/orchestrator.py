import json
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
