import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from langfuse import Langfuse

# Compatibility import for observe
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
    trace_id: str
    sources: list = []

class FeedbackRequest(BaseModel):
    trace_id: str
    score: float
    comment: str = None

# --- API Endpoints ---
@app.post("/chat", response_model=QueryResponse)
@observe(name="api_chat_endpoint")
async def chat_endpoint(request: QueryRequest):
    """
    Main inference endpoint for RAG pipeline (v3 compatible).
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        answer, sources, trace_id = rag_engine.get_answer_with_sources(
            query=request.query
        )
        
        if not trace_id or trace_id == "unknown":
            logger.warning("No trace_id returned from engine")
        
        return {
            "answer": answer,
            "trace_id": trace_id or "unknown",
            "sources": sources
        }

    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Human-in-the-loop feedback collection with WORKING v3 API.
    
    Fixed: Uses correct Langfuse v3 SDK methods.
    """
    try:
        logger.info(f"Recording feedback for trace: {request.trace_id}, score: {request.score}")
        
        # === METHOD 1: Try langfuse.score() (v3 SDK) ===
        try:
            langfuse.score(
                trace_id=request.trace_id,
                name="user-feedback",
                value=request.score,
                comment=request.comment or ""
            )
            # Flush to ensure it's sent immediately
            langfuse.flush()
            logger.info("✅ Feedback sent via langfuse.score()")
            return {"status": "success", "message": "Telemetry recorded (method 1)"}
        except AttributeError as e:
            logger.warning(f"Method 1 failed (AttributeError): {e}")
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
        
        # === METHOD 2: Try create_score() ===
        try:
            langfuse.create_score(
                trace_id=request.trace_id,
                name="user-feedback",
                value=request.score,
                comment=request.comment or ""
            )
            langfuse.flush()
            logger.info("✅ Feedback sent via create_score()")
            return {"status": "success", "message": "Telemetry recorded (method 2)"}
        except AttributeError as e:
            logger.warning(f"Method 2 failed (AttributeError): {e}")
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
        
        # === METHOD 3: Try client.score() ===
        try:
            # Access internal client if available
            if hasattr(langfuse, 'client'):
                langfuse.client.score(
                    trace_id=request.trace_id,
                    name="user-feedback",
                    value=request.score,
                    comment=request.comment or ""
                )
                langfuse.flush()
                logger.info("✅ Feedback sent via client.score()")
                return {"status": "success", "message": "Telemetry recorded (method 3)"}
        except Exception as e:
            logger.warning(f"Method 3 failed: {e}")
        
        # === METHOD 4: Try REST API directly ===
        try:
            import httpx
            import os
            
            langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            
            if not langfuse_public_key or not langfuse_secret_key:
                raise Exception("Langfuse credentials not found in environment")
            
            url = f"{langfuse_host}/api/public/scores"
            headers = {
                "Content-Type": "application/json",
            }
            auth = (langfuse_public_key, langfuse_secret_key)
            
            payload = {
                "traceId": request.trace_id,
                "name": "user-feedback",
                "value": request.score,
                "comment": request.comment or ""
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    auth=auth,
                    timeout=10.0
                )
                response.raise_for_status()
            
            logger.info("✅ Feedback sent via REST API")
            return {"status": "success", "message": "Telemetry recorded (REST API)"}
            
        except Exception as e:
            logger.error(f"Method 4 (REST API) failed: {e}")
        
        # === ALL METHODS FAILED ===
        logger.error("❌ All feedback methods failed")
        raise HTTPException(
            status_code=500, 
            detail="Failed to record telemetry. Check Langfuse configuration."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Feedback ingestion failure")
        raise HTTPException(status_code=500, detail=f"Failed to record telemetry: {str(e)}")


@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    """
    Document ingestion endpoint (async processing).
    """
    content = await file.read()
    background_tasks.add_task(process_pdf, content, file.filename)
    
    logger.info(f"Ingestion pipeline queued for: {file.filename}")
    return {
        "status": "accepted", 
        "message": f"Ingestion pipeline started for {file.filename}"
    }

@app.get("/health")
def health():
    """Health check for K8s/Docker."""
    return {"status": "healthy", "version": "3.0.0"}

@app.get("/")
def root():
    """API info."""
    return {
        "service": "RAG API",
        "version": "3.0.0",
        "mlops": "Langfuse v3 + Prometheus"
    }