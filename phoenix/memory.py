from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from config import CONVERSATION_DB_PATH, CHROMA_DIR, EMBEDDING_MODEL, MAX_WORKING_MEMORY_TOKENS

def get_chat_message_history(session_id="default"):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{CONVERSATION_DB_PATH}"
    )

def create_working_memory(llm, session_id="default"):
    history = get_chat_message_history(session_id)
    return ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=history,
        max_token_limit=MAX_WORKING_MEMORY_TOKENS,
        return_messages=True,
        memory_key="chat_history",
        input_key="input"
    )

class SemanticMemoryStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
            collection_name="phoenix_semantic_memory"
        )

    def add_memory(self, text, metadata=None):
        doc = Document(page_content=text, metadata=metadata or {})
        self.vectorstore.add_documents([doc])
        self.vectorstore.persist()

    def recall(self, query, k=3):
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def forget(self, query, k=1):
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        if docs:
            ids_to_delete = [self.vectorstore._collection.id_to_uuid[doc.metadata["id"]] for doc, _ in docs]
            self.vectorstore._collection.delete(ids=ids_to_delete)
            self.vectorstore.persist()
