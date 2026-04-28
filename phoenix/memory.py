import sqlite3
import datetime
import logging
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import CONVERSATION_DB_PATH, CHROMA_DIR, EMBEDDING_MODEL, MAX_WORKING_MEMORY_TOKENS

logger = logging.getLogger("phoenix.memory")

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
        # 🔧 Fix: LlamaCpp returns a string, not an object with .content
        summary_text = summary_response.strip() if isinstance(summary_response, str) else summary_response.content.strip()
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
        return doc

    def add_improvement_log(self, topic, summary, source_url=None):
        metadata = {"type": "improvement_log"}
        if source_url:
            metadata["source_url"] = source_url
        text = f"IMPROVEMENT: {topic}\nSummary: {summary}"
        self.add_memory(text, metadata=metadata)

    def recall(self, query, k=3):
        return multi_signal_recall(query, self.vectorstore, k)

    def verify_citation(self, doc_id):
        return True

    def get_all_memories(self, limit=100):
        collection = self.vectorstore._collection
        results = collection.get(limit=limit)
        return results['documents'] if results else []

    def delete_memory_by_id(self, doc_id):
        self.vectorstore._collection.delete(ids=[doc_id])

    def find_duplicates(self, threshold=0.95):
        # Placeholder – for now returns empty list
        return []
