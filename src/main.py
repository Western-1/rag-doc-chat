import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# --- MLOps: Observability imports ---
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# --- FIX: Import the instantiated 'engine' directly ---
from src.rag import engine as rag_engine
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
# Initialize the global Langfuse client for scoring/feedback
langfuse = Langfuse()

# --- Data Models ---

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    trace_id: str
    sources: list = []

class FeedbackRequest(BaseModel):
    trace_id: str
    score: float
    comment: str = None

# --- API Endpoints ---

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Executes the full RAG pipeline:
    1. Multi-query generation
    2. Deep vector retrieval
    3. Cross-encoder reranking
    4. Answer generation with Langfuse tracing
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        # Create a fresh callback handler for this specific request to track the trace
        langfuse_handler = CallbackHandler()
        
        # Invoke the engine
        answer, sources, trace_id = rag_engine.get_answer_with_sources(
            query=request.query, 
            callbacks=[langfuse_handler]
        )
        
        # Fallback if trace_id wasn't captured by the handler
        if not trace_id and langfuse_handler.get_trace_id():
            trace_id = langfuse_handler.get_trace_id()
        
        return {
            "answer": answer,
            "trace_id": trace_id or "unknown",
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Inference failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Records human-in-the-loop feedback (scores) to Langfuse.
    This is critical for evaluating RAG performance over time.
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
    Accepts PDF uploads and processes them in the background 
    to prevent blocking the main thread.
    """
    content = await file.read()
    
    background_tasks.add_task(process_pdf, content, file.filename)
    
    return {
        "status": "accepted", 
        "message": f"Ingestion pipeline started for {file.filename}"
    }

@app.get("/health")
def health():
    """Kubernetes/Docker health check endpoint."""
    return {"status": "healthy"}