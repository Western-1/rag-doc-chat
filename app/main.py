import os
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from prometheus_fastapi_instrumentator import Instrumentator  # NEW: for metrics

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAG-Service")

app = FastAPI(title="RAG Production API", version="1.4")  # UPDATED: version

# NEW: Prometheus metrics
Instrumentator().instrument(app).expose(app)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# PERSISTENT DB
db_path = os.path.join(os.getcwd(), "qdrant_db")
os.makedirs(db_path, exist_ok=True)
client = QdrantClient(path=db_path)
collection_name = "pdf_docs"

# MLOps: Create collection if missing
if not any(c.name == collection_name for c in client.get_collections().collections):
    logger.info(f"Creating new collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

# TUNING: Increase k to 10 to find specific entities like prices
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

template = """Answer ONLY based on context. If price ($695) is in text, you MUST find it. Context: {context} Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

add_routes(app, rag_chain, path="/chat")

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Start ingestion for file: {file.filename}")
        reader = PdfReader(file.file)
        text = "".join([p.extract_text() or "" for p in reader.pages])
        # UPDATED: Slightly larger chunks for better context
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(text)
        vector_store.add_texts(chunks)
        logger.info(f"Successfully indexed {len(chunks)} chunks")
        return {"status": "success", "indexed_chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Healthcheck endpoint for K8s probes
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting RAG Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)