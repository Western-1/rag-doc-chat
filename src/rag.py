import logging
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import EMBEDDING_MODEL, LLM_MODEL, QDRANT_URL, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        logger.info(f"Завантаження моделі ембедінгів: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        self.llm = ChatGroq(
            model=LLM_MODEL, 
            temperature=0,
            max_tokens=200, 
            max_retries=3 
        )
        
        self.client = QdrantClient(url=QDRANT_URL)
        self.vector_store = self._init_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def _init_vector_store(self):
        try:
            self.client.get_collection(COLLECTION_NAME)
            logger.info(f"Колекція {COLLECTION_NAME} вже існує.")
        except Exception:
            logger.info(f"Створення нової колекції {COLLECTION_NAME}...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        
        return QdrantVectorStore(
            client=self.client, 
            collection_name=COLLECTION_NAME, 
            embedding=self.embeddings
        )

    def get_prompt(self):
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert technical assistant. Answer accurately using ONLY the context provided.\n"
                "RULES: 1. Full sentences. 2. Be factual. 3. No outside knowledge. 4. If unsure, say 'I don't know'."
            ),
            HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
        ])

    def get_chain(self):
        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)
        
        return (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()} 
            | self.get_prompt() 
            | self.llm 
            | StrOutputParser()
        )


engine = RAGEngine()

def get_rag_chain():
    """
    Цю функцію викликає main.py. Вона повертає скомпільований ланцюжок.
    """
    return engine.get_chain()