import logging
import hashlib
import re
from typing import List, Optional, Dict, Tuple

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.messages import HumanMessage, AIMessage
from flashrank import Ranker, RerankRequest

# --- Langfuse client + compatible observe import ---
from langfuse import Langfuse
try:
    from langfuse import observe
except Exception:
    try:
        from langfuse.decorators import observe
    except Exception:
        def observe(name=None):
            def _decorator(fn):
                return fn
            return _decorator

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    EMBEDDING_MODEL, 
    LLM_MODEL, 
    QDRANT_URL, 
    COLLECTION_NAME, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        logger.info(f"--- INITIALIZING RAG ENGINE ({COLLECTION_NAME}) ---")
        self.langfuse = Langfuse()

        # Initialize Embeddings and LLMs
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatGroq(
            model=LLM_MODEL, 
            temperature=0.0,
            max_tokens=1024,
            max_retries=3,
            stop=["<|eot_id|>", "<|start_header_id|>", "assistant<|header_end|>", "<|end_of_text|>"],
            model_kwargs={"frequency_penalty": 0.0, "presence_penalty": 0.0}
        )
        self.query_generator_llm = ChatGroq(model=LLM_MODEL, temperature=0.5)

        # Reranker
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")

        # Qdrant
        self.client = QdrantClient(url=QDRANT_URL)
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> QdrantVectorStore:
        try:
            self.client.get_collection(COLLECTION_NAME)
            logger.info(f"✅ Collection '{COLLECTION_NAME}' exists")
        except Exception:
            logger.info(f"Creating collection '{COLLECTION_NAME}'")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        return QdrantVectorStore(client=self.client, collection_name=COLLECTION_NAME, embedding=self.embeddings)

    def clear_database(self) -> bool:
        try:
            self.client.delete_collection(COLLECTION_NAME)
            logger.info("Collection deleted.")
            self._init_vector_store()
            return True
        except Exception as e:
            logger.error(f"Error clearing DB: {e}")
            return False

    @observe(name="ingest_file")
    def ingest_file(self, file_path: str) -> int:
        logger.info(f"Processing file: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            content = doc.page_content
            content = content.replace("-\n", "")
            content = content.replace("\n", " ")
            content = re.sub(r'\s+', ' ', content)
            doc.page_content = content

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[". ", "? ", "! ", "; ", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        for doc in splits:
            doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc.metadata["doc_hash"] = doc_hash

        if splits:
            self.vector_store.add_documents(splits)
            logger.info(f"✅ Indexed {len(splits)} chunks")

        return len(splits)

    @observe(name="generate_multi_queries")
    def _generate_multi_queries(self, original_query: str) -> List[str]:
        QUERY_PROMPT = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Generate 3 different search queries based on the user question. Return only the queries, one per line."),
            ("human", "{question}"),
        ])
        chain = QUERY_PROMPT | self.query_generator_llm | StrOutputParser()
        try:
            response = chain.invoke({"question": original_query})
            queries = [line.strip() for line in response.split("\n") if line.strip()]
            return list(set(queries + [original_query]))
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return [original_query]

    @observe(name="rerank_docs")
    def rerank_docs(self, query: str, docs: List) -> List:
        if not docs:
            return []
        unique_docs = []
        seen_hashes = set()
        for doc in docs:
            doc_hash = doc.metadata.get("doc_hash") or hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()
            if doc_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(doc_hash)
        if not unique_docs:
            unique_docs = docs

        passages = [{"id": str(i), "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(unique_docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        return results[:7]

    @observe(name="rag_pipeline")
    def get_answer_with_sources(
        self, 
        query: str, 
        chat_history: List[Dict] = None,
        custom_system_prompt: Optional[str] = None
    ) -> Tuple[str, List, str]:
        if chat_history is None:
            chat_history = []

        generated_queries = self._generate_multi_queries(query)
        logger.info(f"🔍 Generated {len(generated_queries)} queries")

        all_docs = []
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 50})
        for q in generated_queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        reranked_results = self.rerank_docs(query, all_docs)
        context_text = "\n\n---\n\n".join([res['text'] for res in reranked_results])
        logger.info(f"📚 Using {len(reranked_results)} chunks as context")

        # Визначаємо промпт
        if custom_system_prompt:
            logger.info("🎨 Using custom system prompt")
            lc_prompt = ChatPromptTemplate.from_messages([
                ("system", custom_system_prompt),
                ("system", "Context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            try:
                if hasattr(self.langfuse, "update_current_trace"):
                    self.langfuse.update_current_trace(tags=["custom-prompt"])
            except Exception:
                pass
        else:
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful QA assistant. Answer based on: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            try:
                langfuse_prompt = self.langfuse.get_prompt("rag-main-prompt")
                lc_prompt = langfuse_prompt.get_langchain_prompt()
                logger.info("✅ Loaded Langfuse prompt")
                try:
                    if hasattr(self.langfuse, "update_current_trace"):
                        self.langfuse.update_current_trace(tags=["managed-prompt"])
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"⚠️ Langfuse prompt failed: {e}")
                lc_prompt = fallback_prompt

        # Підготовка історії чату
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        # Створення та виклик ланцюжка з явною структурою вхідних даних
        input_data = {
            "context": context_text,
            "question": query,
            "chat_history": formatted_history
        }

        try:
            # Створюємо Runnable ланцюжок
            chain = lc_prompt | self.llm | StrOutputParser()
            answer = chain.invoke(input_data)
        except Exception as e:
            logger.error(f"Chain invocation failed: {e}")
            # Фолбек на прямий виклик моделі у разі помилки Runnable
            fallback_msg = f"Context: {context_text}\n\nQuestion: {query}"
            answer = self.llm.predict(fallback_msg)

        # Очищення відповіді
        answer = re.sub(r'【.*?】', '', answer)
        answer = re.sub(r'\[.*?\]', '', answer)

        # Захоплення trace_id
        trace_id = "unknown"
        try:
            if hasattr(self.langfuse, "get_current_trace_id"):
                trace_id = self.langfuse.get_current_trace_id() or "unknown"
                logger.info(f"📊 Trace ID: {trace_id}")
        except Exception as e:
            logger.warning(f"⚠️ Trace ID capture failed: {e}")

        return answer.strip(), reranked_results, trace_id

# Global engine instance
engine = RAGEngine()