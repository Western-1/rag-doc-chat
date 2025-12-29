import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from langserve import add_routes
from prometheus_fastapi_instrumentator import Instrumentator

# --- SMART LANGFUSE IMPORT ---
try:
    from langfuse.langchain import CallbackHandler
except ImportError:
    try:
        from langfuse.callback import CallbackHandler
    except ImportError:
        from langfuse.langchain import LangfuseCallbackHandler as CallbackHandler

from src.rag import get_rag_chain
from src.ingestion import process_pdf

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-API")

app = FastAPI(
    title="Talk To Your Docs - MLOps Edition",
    version="2.0.0"
)

Instrumentator().instrument(app).expose(app)

# --- LANGFUSE SETUP ---
langfuse_handler = CallbackHandler()

# --- LANGSERVE (PLAYGROUND) ---
chain = get_rag_chain().with_config({"callbacks": [langfuse_handler]})

add_routes(
    app, 
    chain, 
    path="/playground"
)

# --- MAIN API ENDPOINT ---
class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        response = chain.invoke(request.query)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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