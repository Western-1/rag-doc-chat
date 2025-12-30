import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from langserve import add_routes
from prometheus_fastapi_instrumentator import Instrumentator

# MLOps: Observability imports
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

try:
    from langfuse.callback import CallbackHandler
except ImportError:
    from langfuse.langchain import CallbackHandler

from src.rag import get_rag_chain
from src.ingestion import process_pdf

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-API")

app = FastAPI(
    title="Talk To Your Docs - MLOps Edition",
    version="2.1.0"
)

# --- MLOps: Metrics (Prometheus) ---
Instrumentator().instrument(app).expose(app)

# --- MLOps: Tracing Setup ---
langfuse = Langfuse()
langfuse_handler = CallbackHandler()

chain = get_rag_chain()

# --- MLOps: Interactive Playground ---
add_routes(
    app, 
    chain.with_config({"callbacks": [langfuse_handler]}), 
    path="/playground",
    enable_feedback_endpoint=True 
)

# --- Data Models ---

class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    trace_id: str
    score: float
    comment: str = None

# --- API Endpoints ---

@app.post("/chat")
@observe(name="chat-endpoint")
async def chat_endpoint(request: QueryRequest):
    """
    Executes RAG logic with automatic tracing.
    """
    try:
        logger.info(f"Query: {request.query}")
        
        langchain_handler = langfuse_context.get_current_langchain_handler()
        
        response = chain.invoke(request.query, config={"callbacks": [langchain_handler]})
        
        current_trace_id = langfuse_context.get_current_trace_id()
        
        return {
            "answer": response,
            "trace_id": current_trace_id 
        }
    except Exception as e:
        logger.error(f"Inference failure: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Records human-in-the-loop feedback.
    FIX: Uses direct client instead of context for async scoring.
    """
    try:
        langfuse.score(
            trace_id=request.trace_id,
            name="user-feedback",
            value=request.score,
            comment=request.comment
        )
        return {"status": "success", "message": "Telemetry recorded"}
    except Exception as e:
        logger.error(f"Feedback ingestion failure: {e}")
        raise HTTPException(status_code=500, detail="Failed to record telemetry")

@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    """
    Asynchronous ingestion.
    """
    content = await file.read()
    background_tasks.add_task(process_pdf, content, file.filename)
    
    return {
        "status": "accepted", 
        "message": f"Ingestion pipeline started for {file.filename}"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}