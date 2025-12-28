import os
import logging
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Service")

app = FastAPI(
    title="Talk To Your Docs - RAG API",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
logger.info(f"Connecting to Qdrant at: {QDRANT_URL}")

client = QdrantClient(url=QDRANT_URL)

COLLECTION_NAME = "documents"

existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

prompt = ChatPromptTemplate.from_template(
    """Answer ONLY from the provided context.

Context:
{context}

Question:
{question}
"""
)

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

add_routes(app, rag_chain, path="/chat")

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = splitter.split_text(text)
        vector_store.add_texts(chunks)

        logger.info(f"Indexed {len(chunks)} chunks")
        return {"status": "ok", "chunks": len(chunks)}

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
