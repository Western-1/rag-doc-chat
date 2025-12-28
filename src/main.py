import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from langserve import add_routes
from prometheus_fastapi_instrumentator import Instrumentator

from src.rag import get_rag_chain
from src.ingestion import process_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-API")

app = FastAPI(
    title="Talk To Your Docs - MLOps Edition",
    version="2.0.0"
)

Instrumentator().instrument(app).expose(app)

rag_chain = get_rag_chain()
add_routes(app, rag_chain, path="/chat")

@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    content = await file.read()
    
    background_tasks.add_task(process_pdf, content, file.filename)
    
    return {
        "status": "accepted", 
        "message": f"File {file.filename} queued for processing"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}