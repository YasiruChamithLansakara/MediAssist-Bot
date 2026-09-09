#!/usr/bin/env bash
#
# MediAssist — Linux environment setup
# ====================================
# Builds .venv/ from scratch and verifies every runtime component.
#
#   ./setup_linux.sh              # full install
#   ./setup_linux.sh --check      # verify an existing .venv, install nothing
#
# The repository previously carried a Windows venv (venv/Scripts/*.exe) which
# cannot run on Linux at all. This script creates .venv/ alongside it; both
# are gitignored, and nothing here touches the Windows one.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_ROOT/.venv"
PY="${PYTHON:-python3}"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- interpreter
if [[ $CHECK_ONLY -eq 0 ]]; then
  say "Python interpreter"
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "  $PY not found. Install Python 3.11+ or set PYTHON=/path/to/python3." >&2
    exit 1
  fi
  "$PY" --version

  say "Creating virtualenv at .venv"
  [[ -d "$VENV" ]] && warn "reusing existing $VENV"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel
  ok "pip $("$VENV/bin/pip" --version | awk '{print $2}')"

  # ------------------------------------------------------------------ install
  say "Core dependencies"
  "$VENV/bin/pip" install -q -r "$PROJECT_ROOT/requirements.txt"
  ok "API, drug lookup, OCR wrapper, tests"

  # CPU-only torch FIRST. The default PyPI torch drags in ~3 GB of CUDA
  # wheels that this project never uses — easyocr runs gpu=False and the
  # embedding model runs on CPU.
  say "PyTorch (CPU build)"
  "$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
  ok "$("$VENV/bin/python" -c 'import torch; print("torch", torch.__version__)')"

  say "Retrieval + second OCR engine"
  "$VENV/bin/pip" install -q -r "$PROJECT_ROOT/requirements-rag.txt"
  ok "faiss, sentence-transformers, rank-bm25, easyocr"

  say "LLM client"
  "$VENV/bin/pip" install -q -r "$PROJECT_ROOT/requirements-llm.txt"
  ok "groq, openai"

  say "spaCy (optional NER support)"
  "$VENV/bin/pip" install -q -r "$PROJECT_ROOT/requirements-nlp.txt" || \
    warn "spaCy install failed — the rule-based extractor still works"
fi

# ------------------------------------------------------------------- verify
say "Verifying runtime"
"$VENV/bin/python" - <<'PYCHECK'
import importlib, shutil, sys

REQUIRED = ["fastapi", "pandas", "rapidfuzz", "pytesseract", "PIL"]
OPTIONAL = {
    "faiss":                 "semantic search disabled",
    "sentence_transformers": "embeddings fall back to TF-IDF (not semantic)",
    "rank_bm25":             "hybrid search degrades to dense-only",
    "easyocr":               "OCR falls back to Tesseract only",
    "groq":                  "chat answers fall back to rule-based",
    "spacy":                 "NER runs rule-based only",
}

missing = []
for mod in REQUIRED:
    try:
        importlib.import_module(mod)
        print(f"  \033[32m✓\033[0m {mod}")
    except ImportError:
        missing.append(mod)
        print(f"  \033[31m✗\033[0m {mod}  (REQUIRED)")

for mod, consequence in OPTIONAL.items():
    try:
        importlib.import_module(mod)
        print(f"  \033[32m✓\033[0m {mod}")
    except ImportError:
        print(f"  \033[33m!\033[0m {mod}  — {consequence}")

# scispacy cannot install on Python 3.13+; say so rather than look broken.
try:
    importlib.import_module("scispacy")
    print("  \033[32m✓\033[0m scispacy")
except ImportError:
    if sys.version_info >= (3, 13):
        print(f"  \033[33m!\033[0m scispacy — unavailable on Python "
              f"{sys.version_info.major}.{sys.version_info.minor} "
              f"(pins spacy<3.8, whose thinc has no wheel). See requirements-nlp.txt.")
    else:
        print("  \033[33m!\033[0m scispacy — not installed (optional)")

if shutil.which("tesseract"):
    print("  \033[32m✓\033[0m tesseract binary")
else:
    print("  \033[33m!\033[0m tesseract binary NOT found — install it for the "
          "printed-text OCR path:")
    print("      Debian/Ubuntu: sudo apt install tesseract-ocr")
    print("      Fedora:        sudo dnf install tesseract")

if missing:
    print(f"\n\033[31mMissing required packages: {', '.join(missing)}\033[0m")
    raise SystemExit(1)
PYCHECK

# --------------------------------------------------------------------- config
say "Configuration"
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  ok "created .env from .env.example — add your LLM_API_KEY"
else
  ok ".env present"
fi

if grep -q "llama-3.1-8b-instant" "$PROJECT_ROOT/.env" 2>/dev/null; then
  warn "your .env pins llama-3.1-8b-instant, decommissioned 2026-08-16."
  warn "set LLM_MODEL=openai/gpt-oss-20b"
fi

cat <<EOF

$(printf '\033[1mReady.\033[0m')

  source .venv/bin/activate
  uvicorn app.main:app --reload          # API on http://127.0.0.1:8000
  pytest -q                              # test suite
  python scripts/evaluate_pipeline.py    # component scores

  cd mediassist-frontend && npm install && npm run dev

The vector index rebuilds automatically on first startup (~1 min for 3,881
drugs). Delete data/faiss_index/ any time the drug CSV changes.
EOF
