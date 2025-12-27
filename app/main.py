import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
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

load_dotenv()

app = FastAPI(title="RAG Production API", version="1.3")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# PERSISTENT DB
db_path = os.path.join(os.getcwd(), "qdrant_db")
os.makedirs(db_path, exist_ok=True)
client = QdrantClient(path=db_path)
collection_name = "pdf_docs"

# MLOps: Create collection if missing
if not any(c.name == collection_name for c in client.get_collections().collections):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

# TUNING: Increase k to 10 to find specific entities like prices
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

template = """Answer ONLY based on context. 
If price ($695) is in text, you MUST find it.
Context: {context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

add_routes(app, rag_chain, path="/chat")

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    
    # TUNING: Smaller chunks for better entity extraction
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_text(text)
    
    vector_store.add_texts(chunks)
    return {"status": "success", "indexed_chunks": len(chunks)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)