import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import (
    FastAPI,
    Query,
    HTTPException,
    APIRouter,
    Request,
    UploadFile,
    File,
    Form,
    Path,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

from app.services.drug_lookup import init_store, lookup_drug, SUPPORTED_DISEASES
from app.services.chat_service import build_chat_response
from app.services.ocr_service import (
    OCRDependencyError,
    OCRImageError,
    analyze_prescription_text,
    ocr_prescription_image,
    ocr_runtime_status,
)
from app.services.conversation_memory import (
    get_conversation_memory,
    add_turn_to_memory,
    get_conversation_history,
    get_context_summary,
)
from app.services.llm_service import (
    get_llm_service,
    is_llm_available,
    generate_llm_response,
    get_rag_status,
)

# Optional RAG service
try:
    from app.services.rag_vector_search import (
        get_rag_service,
        is_rag_available,
        vectorize_drug_knowledge,
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    get_rag_service = None
    is_rag_available = lambda: False
    vectorize_drug_knowledge = None


# -----------------------------------------------------------------------------
# ENV / CONFIG
# -----------------------------------------------------------------------------
ENV = os.getenv("ENV", "development").strip().lower()  # development | production
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

RATE_LIMIT_ENABLED = os.getenv(
    "RATE_LIMIT_ENABLED", "0" if ENV == "development" else "1"
).strip() == "1"

RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))  # requests per minute per IP
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))  # burst allowance
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))


def _get_allowed_origins() -> list[str]:
    """
    Comma-separated origins, e.g.
    ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
    """
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()

    # production-safe: if not set, disable CORS
    if ENV == "production" and not raw:
        logging.warning("ALLOWED_ORIGINS not set in production; CORS will be disabled.")
        return []

    if not raw:
        raw = "http://localhost:5173,http://127.0.0.1:5173"

    return [o.strip() for o in raw.split(",") if o.strip()]


def _normalize_disease(d: str) -> str:
    return " ".join((d or "").strip().lower().split())


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mediassist")


# -----------------------------------------------------------------------------
# MIDDLEWARE: request_id + logging + rate limiting
# -----------------------------------------------------------------------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = int((time.time() - start) * 1000)
            client_ip = (
                request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or (request.client.host if request.client else "unknown")
            )
            status_code = getattr(response, "status_code", "-")
            logger.info(
                "%s %s -> %s (%dms) ip=%s rid=%s",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                client_ip,
                request_id,
            )

            if response is not None:
                response.headers["X-Request-ID"] = request_id


class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """
    In-memory per-IP rate limiter.
    Good for demo/single-instance.
    For multi-instance production, use Redis-based limiter.
    """
    def __init__(self, app: Any, rpm: int, burst: int):
        super().__init__(app)
        self.rpm = max(1, int(rpm))
        self.burst = max(0, int(burst))
        self.window_seconds = 60
        self._hits: dict[str, list[float]] = {}
        self._lock = __import__("threading").Lock()

    def _client_ip(self, request: Request) -> str:
        xf = request.headers.get("x-forwarded-for", "")
        if xf:
            return xf.split(",")[0].strip() or "unknown"
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        # allow health/meta without limiting
        if request.url.path.endswith("/health") or request.url.path.endswith("/meta"):
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.time()
        cutoff = now - self.window_seconds
        allowed = self.rpm + self.burst

        with self._lock:
            arr = self._hits.get(ip, [])
            arr = [t for t in arr if t >= cutoff]  # prune
            if len(arr) >= allowed:
                rid = getattr(request.state, "request_id", "")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "rate_limited",
                            "message": "Too many requests. Please slow down.",
                            "details": {"rpm": self.rpm, "window_seconds": self.window_seconds},
                        },
                        "request_id": rid,
                    },
                    headers={"X-Request-ID": rid} if rid else None,
                )
            arr.append(now)
            self._hits[ip] = arr

        return await call_next(request)


# -----------------------------------------------------------------------------
# APP LIFESPAN
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize drug lookup store
    init_store()
    logger.info("Drug lookup store initialized")
    
    # Initialize RAG if available
    if RAG_AVAILABLE and is_rag_available and is_rag_available():
        try:
            rag_service = get_rag_service()
            if rag_service and rag_service.is_available():
                # Get drug data for vectorization
                from app.services.drug_lookup import _df, _index
                
                if _df is not None and len(_df) > 0:
                    # Convert dataframe to drug list format
                    drugs = []
                    for _, row in _df.iterrows():
                        drug_entry = {
                            "name": row.get("brand_name", row.get("generic_name", "Unknown")),
                            "generic_name": row.get("generic_name", "Unknown"),
                            "sections": {
                                "indications": row.get("indications", ""),
                                "warnings": row.get("warnings", ""),
                                "contraindications": row.get("contraindications", ""),
                            },
                            "disease": row.get("disease_category", ""),
                        }
                        drugs.append(drug_entry)
                    
                    count = rag_service.vectorize_drugs(drugs)
                    logger.info(f"RAG vectorization complete: {count} drugs indexed")
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
    
    yield


app = FastAPI(title="MediAssist API", version="0.3.0", lifespan=lifespan)

# middleware order
app.add_middleware(RequestContextMiddleware)
if RATE_LIMIT_ENABLED:
    app.add_middleware(SimpleRateLimitMiddleware, rpm=RATE_LIMIT_RPM, burst=RATE_LIMIT_BURST)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# STRUCTURED ERROR HANDLERS
# -----------------------------------------------------------------------------
def _err(request: Request, code: str, message: str, details: Any = None, status_code: int = 400):
    rid = getattr(request.state, "request_id", "")
    payload = {"error": {"code": code, "message": message, "details": details}, "request_id": rid}
    return JSONResponse(status_code=status_code, content=payload, headers={"X-Request-ID": rid} if rid else None)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _err(
        request,
        code="validation_error",
        message="Request validation failed",
        details=exc.errors(),
        status_code=422,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "http_error")
        message = detail.get("message", "Request failed")
        details = {k: v for k, v in detail.items() if k not in {"code", "message"}}
        return _err(request, code=code, message=message, details=details or None, status_code=exc.status_code)
    return _err(request, code="http_error", message=str(detail), details=None, status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return _err(request, code="internal_error", message="Internal server error", details=None, status_code=500)


# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
api = APIRouter(prefix="/api", tags=["api"])


@api.get("/health")
def health():
    return {"status": "ok"}


# optional compatibility route (keeps old /health checks working)
@app.get("/health")
def health_root():
    return {"status": "ok"}


@api.get("/meta")
def meta():
    return {
        "supported_diseases": SUPPORTED_DISEASES,
        "age_range": {"min": 1, "max": 120},
        "env": ENV,
        "rate_limit": {"enabled": RATE_LIMIT_ENABLED, "rpm": RATE_LIMIT_RPM, "burst": RATE_LIMIT_BURST},
        "features": {"chat": True, "prescription_ocr": True, "lightweight_ner": True},
        "ocr_runtime": ocr_runtime_status(),
    }


def _validate_context(disease: str, age: int):
    d = _normalize_disease(disease)
    if d not in set(SUPPORTED_DISEASES):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_disease",
                "message": "Unsupported disease",
                "supported_diseases": SUPPORTED_DISEASES,
            },
        )
    if age < 1 or age > 120:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_age", "message": "Age must be 1..120"},
        )
    return d


@api.get("/lookup")
def lookup(
    request: Request,
    drug: str = Query(..., min_length=1),
    disease: str = Query(...),
    age: int = Query(..., ge=1, le=120),
):
    q = (drug or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail={"code": "empty_query", "message": "Drug must not be empty."})

    d = _validate_context(disease, age)

    result = lookup_drug(q, disease=d, age=age)
    rid = getattr(request.state, "request_id", "")
    if rid:
        result["request_id"] = rid
    return result


# -----------------------------------------------------------------------------
# CHAT: POST /api/chat
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    disease: str = Field(..., description="One of supported diseases")
    age: int = Field(..., ge=1, le=120)
    message: str = Field(..., min_length=1, description="User question/message")
    drug: Optional[str] = Field(None, description="Optional single drug name")
    drugs: Optional[list[str]] = Field(None, description="Optional list of drug names")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation memory")


class PrescriptionTextRequest(BaseModel):
    disease: str = Field(..., description="One of supported diseases")
    age: int = Field(..., ge=1, le=120)
    text: str = Field(..., min_length=1, max_length=20000, description="OCR text or manually corrected prescription text")


@api.post("/chat")
def chat(request: Request, payload: ChatRequest):
    d = _validate_context(payload.disease, payload.age)

    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(status_code=422, detail={"code": "empty_message", "message": "Message must not be empty."})

    # normalize optional drug(s)
    drugs_in = []
    if payload.drug:
        drugs_in.append(payload.drug)
    if payload.drugs:
        drugs_in.extend(payload.drugs)

    drugs_in = [str(x).strip() for x in drugs_in if str(x).strip()]
    rid = getattr(request.state, "request_id", "")
    
    # Use or generate session ID
    session_id = (payload.session_id or "").strip()
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get conversation history if available
    conversation_history = get_conversation_history(session_id, max_turns=10)

    # Build initial chat response (rule-based)
    response = build_chat_response(
        message=msg,
        disease=d,
        age=int(payload.age),
        drugs=drugs_in,
        request_id=rid,
    )
    
    # Try to enhance with LLM if available
    llm_service = get_llm_service()
    if llm_service.is_available() and response.get("matched_drugs"):
        # Filter to matched drugs only
        matched_drugs = [m for m in response.get("matched_drugs", []) if m.get("best_match")]
        
        if matched_drugs:
            llm_response = llm_service.generate_response(
                message=msg,
                disease=d,
                age=int(payload.age),
                matched_drugs=matched_drugs,
                conversation_history=conversation_history,
            )
            
            if llm_response:
                response["answer"] = llm_response
                response["answer_source"] = "llm_grounded"
            else:
                response["answer_source"] = "rule_based"
        else:
            response["answer_source"] = "rule_based"
    else:
        response["answer_source"] = "rule_based"
    
    response["llm_available"] = llm_service.is_available()
    response["session_id"] = session_id
    
    # Store in conversation memory
    add_turn_to_memory(
        session_id=session_id,
        role="user",
        text=msg,
        data={"context": payload.dict()},
        disease=d,
        age=int(payload.age),
    )
    
    add_turn_to_memory(
        session_id=session_id,
        role="assistant",
        text=response.get("answer", ""),
        data=response,
        disease=d,
        age=int(payload.age),
    )
    
    return response


# -----------------------------------------------------------------------------
# PRESCRIPTION OCR: POST /api/prescription
# multipart/form-data: file + disease + age
# -----------------------------------------------------------------------------
@api.post("/prescription")
async def prescription_ocr(
    request: Request,
    file: UploadFile = File(...),
    disease: str = Form(...),
    age: int = Form(...),
):
    d = _validate_context(disease, int(age))

    if not file:
        raise HTTPException(status_code=422, detail={"code": "missing_file", "message": "File is required."})

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail={"code": "empty_file", "message": "Uploaded file is empty."})
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "file_too_large", "message": f"Image must be {MAX_UPLOAD_BYTES} bytes or smaller."},
        )

    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_file_type", "message": "Upload a prescription image file."},
        )

    rid = getattr(request.state, "request_id", "")

    try:
        result = ocr_prescription_image(
            image_bytes=content,
            filename=file.filename or "upload",
            disease=d,
            age=int(age),
            request_id=rid,
        )
    except OCRDependencyError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "ocr_unavailable", "message": str(exc)},
        ) from exc
    except OCRImageError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_image", "message": str(exc)},
        ) from exc
    return result


@api.post("/prescription/analyze-text")
def prescription_analyze_text(request: Request, payload: PrescriptionTextRequest):
    d = _validate_context(payload.disease, payload.age)
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail={"code": "empty_text", "message": "Text must not be empty."})

    rid = getattr(request.state, "request_id", "")
    return analyze_prescription_text(
        text=text,
        disease=d,
        age=int(payload.age),
        request_id=rid,
    )


# CONVERSATION MEMORY: GET /api/chat/history/{session_id}
# Retrieves conversation history for a session
@api.get("/chat/history/{session_id}")
def get_chat_history(request: Request, session_id: str = Path(..., min_length=1)):
    """Get conversation history for a session."""
    history = get_conversation_history(session_id, max_turns=50)
    context = get_context_summary(session_id)
    
    rid = getattr(request.state, "request_id", "")
    return {
        "session_id": session_id,
        "history": history,
        "context": context,
        "request_id": rid if rid else None,
    }


# CONVERSATION MEMORY: GET /api/memory/stats
# Returns conversation memory statistics
@api.get("/memory/stats")
def get_memory_stats(request: Request):
    """Get conversation memory statistics."""
    memory = get_conversation_memory()
    stats = memory.stats()
    
    rid = getattr(request.state, "request_id", "")
    return {
        "stats": stats,
        "llm_available": is_llm_available(),
        "rag_status": get_rag_status(),
        "request_id": rid if rid else None,
    }


# Returns RAG vector database status
@api.get("/rag/status")
def get_rag_db_status(request: Request):
    """Get RAG vector database status."""
    rid = getattr(request.state, "request_id", "")
    return {
        "rag_status": get_rag_status(),
        "request_id": rid if rid else None,
    }


# RAG semantic search
@api.post("/rag/search")
def rag_search(request: Request, query: str = Query(..., min_length=1, max_length=500), top_k: int = Query(5, ge=1, le=20)):
    """
    Perform semantic search on vectorized drug knowledge.
    
    Args:
        query: Search query (drug name, symptom, condition)
        top_k: Number of results to return
    """
    rid = getattr(request.state, "request_id", "")
    
    if not (is_rag_available and is_rag_available()):
        return _err(
            request,
            code="rag_unavailable",
            message="RAG search not available",
            status_code=503,
        )
    
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return _err(
                request,
                code="rag_unavailable",
                message="RAG service initialization failed",
                status_code=503,
            )
        
        results = rag_service.retrieve_context(query, top_k=top_k)
        
        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "count": len(results),
            "request_id": rid if rid else None,
        }
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return _err(
            request,
            code="rag_search_error",
            message=f"RAG search failed: {str(e)}",
            status_code=500,
        )


app.include_router(api)
