# =============================================================================
# MediAssist — production image
# =============================================================================
# One container serves both the API and the built UI, so the browser talks to
# a single origin and CORS disappears from the deployment entirely.
#
# Three things are done at BUILD time so the first user request is fast and
# the running container needs no outbound network:
#   * the sentence-transformers model is downloaded into the image
#   * the FAISS + BM25 retrieval index is built from the drug dataset
#   * the frontend is compiled to static files
#
# Build:  docker build -t mediassist .
# Run:    docker compose up -d
# =============================================================================

# ---------------------------------------------------------------- frontend ---
FROM node:22-slim AS frontend

WORKDIR /ui
COPY mediassist-frontend/package*.json ./
RUN npm ci --no-audit --no-fund

COPY mediassist-frontend/ ./
RUN npm run build


# ------------------------------------------------------------------ python ---
# 3.14 matches the interpreter the pinned requirements were resolved and
# tested against, so the image installs exactly the versions that were
# verified. (SciSpaCy cannot build on 3.14 — it is optional and unused; see
# requirements-nlp.txt.)
FROM python:3.14-slim AS runtime

# tesseract          — the printed-text OCR engine (pytesseract is only a wrapper)
# libgomp1           — OpenMP runtime required by faiss and torch
# libglib2.0-0       — required by opencv, which easyocr depends on
# curl               — container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/models

WORKDIR /app

# CPU-only torch first. The default PyPI wheel pulls ~3 GB of CUDA libraries
# that this image never uses: easyocr runs with gpu=False and the embedding
# model runs on CPU. Installing it first also stops the later requirements
# files from resolving the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0

COPY requirements.txt requirements-rag.txt requirements-llm.txt ./
RUN pip install -r requirements.txt \
    && pip install -r requirements-rag.txt \
    && pip install -r requirements-llm.txt

# Bake the embedding model into the image so the container starts without
# reaching out to HuggingFace — and keeps working if that is ever blocked.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/processed/drug_knowledge_bot_ready_clean.csv ./data/processed/
COPY --from=frontend /ui/dist ./mediassist-frontend/dist

# Pre-build the retrieval index (~16k chunks). Without this the first request
# after every deploy waits about a minute for embeddings.
RUN python -c "\
import app.services.drug_lookup as dl; \
from app.ml.faiss_store import get_faiss_store; \
dl.init_store(); \
s = get_faiss_store(); \
n = s.build(dl._df.to_dict('records')); \
s.save(); \
print(f'prebuilt retrieval index: {n} chunks')"

# Run as a non-root user. The app writes nothing outside /tmp at runtime.
RUN useradd --create-home --uid 10001 mediassist \
    && chown -R mediassist:mediassist /app /opt/models
USER mediassist

ENV ENV=production \
    HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# ONE worker, deliberately. The rate limiter, conversation memory and
# retrieval index are per-process singletons; a second worker would give
# users inconsistent chat history and double the effective rate limit.
# Scale by running more containers behind a load balancer with sticky
# sessions, or move that state to Redis first.
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
