import logging
import hashlib
import re
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.callbacks import BaseCallbackHandler
from flashrank import Ranker, RerankRequest

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import EMBEDDING_MODEL, LLM_MODEL, QDRANT_URL, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Core RAG logic handling document ingestion, vector retrieval, 
    reranking, and response generation with observability hooks.
    """
    def __init__(self):
        logger.info(f"--- INITIALIZING RAG ENGINE ({COLLECTION_NAME}) ---")
        
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize LLM with strict parameters for factual accuracy
        self.llm = ChatGroq(
            model=LLM_MODEL, 
            temperature=0.0, 
            max_tokens=1024,
            max_retries=3,
            stop=["<|eot_id|>", "<|start_header_id|>", "assistant<|header_end|>", "<|end_of_text|>"],
            model_kwargs={"frequency_penalty": 0.0, "presence_penalty": 0.0}
        )

        # Secondary LLM instance for query expansion (higher temperature for creativity)
        self.query_generator_llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.5, 
        )
        
        # Reranker model (FlashRank)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
        
        # Qdrant Client Setup
        self.client = QdrantClient(url=QDRANT_URL)
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> QdrantVectorStore:
        """
        Initializes or retrieves the Qdrant collection.
        """
        try:
            self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        return QdrantVectorStore(
            client=self.client, 
            collection_name=COLLECTION_NAME, 
            embedding=self.embeddings
        )

    def clear_database(self) -> bool:
        """
        Deletes the entire collection from Qdrant.
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            logger.info("Collection deleted.")
            self._init_vector_store()
            return True
        except Exception as e:
            logger.error(f"Error clearing DB: {e}")
            return False

    def ingest_file(self, file_path: str) -> int:
        """
        Processes a PDF file: cleans text, splits into chunks, hashes content for deduplication,
        and uploads to Qdrant. Returns the number of chunks added.
        """
        logger.info(f"Processing file: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Text Cleaning Phase
        for doc in docs:
            content = doc.page_content
            content = content.replace("-\n", "")
            # Replace newlines with spaces to maintain sentence structure
            content = content.replace("\n", " ")
            # Collapse multiple spaces
            content = re.sub(r'\s+', ' ', content)
            doc.page_content = content
        
        # Splitting Phase
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[". ", "? ", "! ", "; ", " ", ""] 
        )
        splits = text_splitter.split_documents(docs)
        
        # Hashing Phase (for potential future deduplication)
        for doc in splits:
            doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc.metadata["doc_hash"] = doc_hash

        if splits:
            self.vector_store.add_documents(splits)
            logger.info(f"Added {len(splits)} chunks to Qdrant.")
        return len(splits)

    def _generate_multi_queries(self, original_query: str) -> List[str]:
        """
        Generates alternative search queries to improve retrieval recall.
        """
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
            logger.warning(f"Query generation failed: {e}. Using original query only.")
            return [original_query]

    def rerank_docs(self, query: str, docs: List) -> List:
        """
        Deduplicates retrieved documents and re-ranks the top results using a cross-encoder.
        """
        if not docs: return []
        
        # Deduplication based on content hash
        unique_docs = []
        seen_hashes = set()
        
        for doc in docs:
            doc_hash = doc.metadata.get("doc_hash")
            if not doc_hash:
                doc_hash = hashlib.md5(doc.page_content.strip().encode('utf-8')).hexdigest()
            
            if doc_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(doc_hash)
        
        if not unique_docs: unique_docs = docs

        # Prepare passages for FlashRank
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(unique_docs)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Return top 7 most relevant chunks
        return results[:7]

    def get_answer_with_sources(self, query: str, callbacks: Optional[List[BaseCallbackHandler]] = None):
        """
        Main RAG pipeline execution:
        1. Multi-query generation
        2. Deep retrieval (k=50)
        3. Reranking
        4. LLM Generation
        5. Output cleanup (regex)
        """
        generated_queries = self._generate_multi_queries(query)
        
        all_docs = []
        # Use high 'k' to cast a wide net for "Deep Retrieval"
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 50} 
        )

        for q in generated_queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        reranked_results = self.rerank_docs(query, all_docs)
        
        context_text = "\n\n---\n\n".join([res['text'] for res in reranked_results])
        
        # Strict system prompt to reduce hallucinations
        system_prompt = """You are a helpful QA assistant. Answer strictly based on the Context provided below.
        
        Rules:
        1. Answer directly and cleanly. Do NOT use citation tags like [1] or 【source】.
        2. If the answer is not in the Context, say "I don't know based on this document."
        3. Quote specific phrases from the text when possible.
        4. Answer in the same language as the Question.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke(
            {"context": context_text, "question": query},
            config={"callbacks": callbacks}
        )
        
        # Regex Post-Processing: Remove model-generated citation artifacts
        answer = re.sub(r'【.*?】', '', answer)
        answer = re.sub(r'\[.*?\]', '', answer)
        answer = answer.strip()

        # Extract Trace ID for feedback loop
        trace_id = None
        if callbacks:
            try:
                trace_id = callbacks[0].get_trace_id()
            except Exception as e:
                logger.warning(f"Could not get trace_id: {e}")

        return answer, reranked_results, trace_id

# --- FIX: Instantiate the global engine object ---
engine = RAGEngine()