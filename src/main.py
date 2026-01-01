import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from langfuse import Langfuse
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-API")

app = FastAPI(title="Talk To Your Docs - MLOps Edition", version="3.0.0")

Instrumentator().instrument(app).expose(app)

# Langfuse client for scoring
langfuse = Langfuse()

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

@app.post("/chat", response_model=QueryResponse)
@observe(name="api_chat")
async def chat_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        answer, sources, trace_id = rag_engine.get_answer_with_sources(
            query=request.query
        )
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
    try:
        # Primary attempt
        try:
            langfuse.score(
                trace_id=request.trace_id,
                name="user-feedback",
                value=request.score,
                comment=request.comment
            )
        except AttributeError:
            # fallback: try client getter if API different
            try:
                from langfuse import get_client
                client = get_client()
                client.score(trace_id=request.trace_id, name="user-feedback", value=request.score, comment=request.comment)
            except Exception as e:
                logger.error(f"Feedback ingestion failure (fallback): {e}")
                raise
        return {"status": "success", "message": "Telemetry recorded"}
    except Exception as e:
        logger.error(f"Feedback ingestion failure: {e}")
        raise HTTPException(status_code=500, detail="Failed to record telemetry")

@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    content = await file.read()
    background_tasks.add_task(process_pdf, content, file.filename)
    return {"status": "accepted", "message": f"Ingestion pipeline started for {file.filename}"}

@app.get("/health")
def health():
    return {"status": "healthy"}
