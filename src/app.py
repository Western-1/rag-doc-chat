import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from langfuse import Langfuse
# Compatibility import for observe (top-level or decorators)
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

from src.rag import engine as rag_engine
from src.ingestion import process_pdf

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-API")

app = FastAPI(
    title="Talk To Your Docs - MLOps Edition",
    version="3.0.0"
)

# --- MLOps: Metrics (Prometheus) ---
Instrumentator().instrument(app).expose(app)

# --- MLOps: Tracing Setup (Langfuse v3) ---
langfuse = Langfuse()

# --- Data Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    trace_id: str  # Used for linking feedback to specific inference runs
    sources: list = []

class FeedbackRequest(BaseModel):
    trace_id: str  # Links back to the original query trace
    score: float   # Typically 0.0 to 1.0 (thumbs down = 0, thumbs up = 1)
    comment: str = None

# --- API Endpoints ---
@app.post("/chat", response_model=QueryResponse)
@observe(name="api_chat_endpoint")  # Wraps the entire API call in a trace (noop if not present)
async def chat_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")

        # Engine returns trace_id if it could capture one
        answer, sources, trace_id = rag_engine.get_answer_with_sources(query=request.query)

        if not trace_id:
            logger.warning("No trace_id returned from engine - feedback loop may be broken")
            trace_id = "unknown"

        return {
            "answer": answer,
            "trace_id": trace_id,
            "sources": sources
        }

    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Human-in-the-loop feedback collection with safe fallback for langfuse API differences.
    """
    try:
        # Primary attempt (typical API)
        try:
            langfuse.score(
                trace_id=request.trace_id,
                name="user-feedback",
                value=request.score,
                comment=request.comment
            )
        except AttributeError:
            # Fallback: try client accessor if top-level 'score' not present
            try:
                from langfuse import get_client
                client = get_client()
                client.score(
                    trace_id=request.trace_id,
                    name="user-feedback",
                    value=request.score,
                    comment=request.comment
                )
            except Exception as e:
                logger.error(f"Feedback ingestion failure (fallback): {e}")
                raise

        logger.info(f"Feedback recorded for trace {request.trace_id}: {request.score}")
        return {"status": "success", "message": "Telemetry recorded"}

    except Exception as e:
        logger.exception("Feedback ingestion failure")
        raise HTTPException(status_code=500, detail="Failed to record telemetry")


@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    content = await file.read()
    background_tasks.add_task(process_pdf, content, file.filename)
    logger.info(f"Ingestion pipeline queued for: {file.filename}")
    return {
        "status": "accepted", 
        "message": f"Ingestion pipeline started for {file.filename}"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {
        "service": "RAG API",
        "version": "3.0.0",
        "mlops": "Langfuse v3 + Prometheus"
    }
