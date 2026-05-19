import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.drug_lookup import init_store, lookup_drug, SUPPORTED_DISEASES
from app.services.chat_service import build_chat_response
from app.services.ocr_service import (
    OCRDependencyError,
    OCRImageError,
    analyze_prescription_text,
    ocr_prescription_image,
    ocr_runtime_status,
)

from app.services.llm_service import (
    get_llm_service,
    get_rag_status,
)

# =========================================
# FAISS RAG IMPORT (NEW ARCHITECTURE)
# =========================================
try:
    from app.services.rag_service import RAGService
    RAG_AVAILABLE = True
except ImportError:
    RAGService = None
    RAG_AVAILABLE = False


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ENV = os.getenv("ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("mediassist")


# -----------------------------------------------------------------------------
# MIDDLEWARE
# -----------------------------------------------------------------------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.time()
        response = await call_next(request)

        duration = int((time.time() - start) * 1000)

        logger.info(
            "%s %s -> %s (%dms) rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response


# -----------------------------------------------------------------------------
# APP LIFESPAN
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init drug store
    init_store()
    logger.info("Drug store initialized")

    # =========================================
    # INIT FAISS RAG SYSTEM
    # =========================================
    if RAG_AVAILABLE:
        try:
            rag = RAGService()

            import pandas as pd
            df = pd.read_csv("data/processed/drug_knowledge_bot_ready_clean.csv")

            drugs = df.to_dict(orient="records")

            rag.vectorize_drugs(drugs)

            app.state.rag_service = rag

            logger.info(f"FAISS RAG initialized with {len(drugs)} drugs")

        except Exception as e:
            logger.warning(f"FAISS RAG init failed: {e}")
            app.state.rag_service = None
    else:
        app.state.rag_service = None
        logger.warning("FAISS RAG not available")

    yield


# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------
app = FastAPI(title="MediAssist API", version="1.0.0", lifespan=lifespan)

app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


# -----------------------------------------------------------------------------
# HEALTH
# -----------------------------------------------------------------------------
@api.get("/health")
def health():
    return {
        "status": "ok",
        "rag": app.state.rag_service is not None,
    }


# -----------------------------------------------------------------------------
# DRUG LOOKUP
# -----------------------------------------------------------------------------
@api.get("/lookup")
def lookup(drug: str, disease: str, age: int):
    return lookup_drug(drug, disease, age)


# -----------------------------------------------------------------------------
# CHAT (UNCHANGED CORE)
# -----------------------------------------------------------------------------
@api.post("/chat")
def chat(payload: dict):
    return build_chat_response(
        message=payload.get("message"),
        disease=payload.get("disease"),
        age=payload.get("age"),
    )


# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------
@api.post("/prescription/analyze")
async def ocr(file: bytes = None):
    try:
        return ocr_prescription_image(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# RAG SEARCH (FAISS POWERED)
# -----------------------------------------------------------------------------
@api.get("/rag/search")
def rag_search(query: str, top_k: int = 5):
    rag = app.state.rag_service

    if not rag:
        return {"error": "RAG not initialized"}

    return {
        "query": query,
        "results": rag.retrieve_context(query, top_k=top_k),
    }


# -----------------------------------------------------------------------------
# ERROR HANDLERS
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(Exception)
async def global_handler(request: Request, exc):
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# -----------------------------------------------------------------------------
# ROUTER INCLUDE
# -----------------------------------------------------------------------------
app.include_router(api)