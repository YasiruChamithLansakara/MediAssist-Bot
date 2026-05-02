# RAG Setup File Locations & Configuration Guide

## 📍 File Locations for Your TF-IDF Configuration

### 1. **Environment Variables File** ✅ CREATED

📄 **File:** `.env.local`  
**Location:** `d:\AI Project\MediAssist-Bot\.env.local`

**Contains your configuration:**

```env
RAG_ENABLED=1
EMBEDDING_PROVIDER=tfidf
USE_CONVERSATION_PERSISTENCE=1
```

**How to use:**

```bash
# Option A: Source the .env.local file (Bash/Linux/Mac)
source .env.local

# Option B: Set variables manually in PowerShell
$env:RAG_ENABLED = "1"
$env:EMBEDDING_PROVIDER = "tfidf"
$env:USE_CONVERSATION_PERSISTENCE = "1"

# Option C: Copy variables to system environment
# (See Windows section below)
```

---

## 🔧 Key Files Where RAG Components Are Integrated

### 2. **RAG Service Core**

📄 **File:** `app/services/rag_vector_search.py`  
**Location:** `d:\AI Project\MediAssist-Bot\app\services\rag_vector_search.py`

**What happens here:**

- Reads `RAG_ENABLED` environment variable (line ~17)
- Reads `EMBEDDING_PROVIDER` to choose TF-IDF (line ~18)
- Initializes TFIDFEmbedding class (line ~170-176)
- Creates SQLite vector database at `VECTOR_DB_PATH` (line ~19)

**Key functions:**

```python
RAG_ENABLED = os.getenv("RAG_ENABLED", "1").lower() in ("1", "true", "yes")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
# Set to "tfidf" for your configuration
```

---

### 3. **Application Main Entry Point**

📄 **File:** `app/main.py`  
**Location:** `d:\AI Project\MediAssist-Bot\app\main.py`

**What happens here:**

- Lines ~25-41: Imports RAG service
- Lines ~195-225: `lifespan()` function initializes RAG on startup
  - Vectorizes all drugs in drug database
  - First run takes ~5-10 seconds
  - Uses TF-IDF embeddings (no external API)
- Lines ~603-614: New `/api/rag/status` endpoint
- Lines ~617-652: New `/api/rag/search` endpoint

**What you'll see in logs:**

```
INFO | Drug lookup store initialized
INFO | RAG vectorization complete: 2847 drugs indexed
INFO | Uvicorn running on http://0.0.0.0:8000
```

---

### 4. **LLM Service (Enhanced)**

📄 **File:** `app/services/llm_service.py`  
**Location:** `d:\AI Project\MediAssist-Bot\app\services\llm_service.py`

**What happens here:**

- Lines ~14-17: Imports RAG service functions
- Lines ~81-130: `build_context_for_llm()` function now calls RAG
- Lines ~112-136: RAG-enhanced context building
- Returns both direct matches and semantic matches

**RAG enhancement flow:**

```
User question → Direct drug lookup
              → RAG semantic search (TF-IDF)
              → LLM gets both sources
              → Better quality response
```

---

### 5. **Conversation Memory**

📄 **File:** `app/services/conversation_memory.py`  
**Location:** `d:\AI Project\MediAssist-Bot\app\services\conversation_memory.py`

**What happens here:**

- Line ~9: Reads `USE_CONVERSATION_PERSISTENCE` environment variable
- Line ~10: Reads `DB_PATH` (default: `data/conversations.db`)
- Lines ~65-100: SQLite database initialization if persistence enabled

**With your configuration:**

```
USE_CONVERSATION_PERSISTENCE=1
↓
Creates: data/conversations.db
↓
Stores conversation history across restarts
```

---

### 6. **Tests**

📄 **File:** `tests/test_rag_service.py`  
**Location:** `d:\AI Project\MediAssist-Bot\tests\test_rag_service.py`

**What happens here:**

- Lines ~6-10: Sets test environment variables
- Tests RAG initialization and search
- Verifies TF-IDF embedding works

---

## 🚀 How to Run

### Step 1: Set Environment Variables

**Option A: Using .env.local (Recommended)**

```bash
# In your IDE terminal or command line
# The application reads from .env.local automatically via python-dotenv
```

**Option B: Manual Setup in PowerShell**

```powershell
# Copy and paste these into your PowerShell terminal:
$env:RAG_ENABLED = "1"
$env:EMBEDDING_PROVIDER = "tfidf"
$env:USE_CONVERSATION_PERSISTENCE = "1"

# Verify they're set:
echo $env:RAG_ENABLED
echo $env:EMBEDDING_PROVIDER
echo $env:USE_CONVERSATION_PERSISTENCE
```

**Option C: Manual Setup in Command Prompt**

```cmd
set RAG_ENABLED=1
set EMBEDDING_PROVIDER=tfidf
set USE_CONVERSATION_PERSISTENCE=1

REM Verify:
echo %RAG_ENABLED%
```

### Step 2: Start Backend

```bash
# From project root directory
cd d:\AI Project\MediAssist-Bot
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**

```
2026-05-01 10:30:00 | INFO | mediassist | Drug lookup store initialized
2026-05-01 10:30:00 | INFO | mediassist.rag | TF-IDF embedding initialized (fallback mode)
2026-05-01 10:30:02 | INFO | mediassist.rag | RAG vectorization complete: 2847 drugs indexed
2026-05-01 10:30:02 | INFO | mediassist | Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Start Frontend (in new terminal)

```bash
cd d:\AI Project\MediAssist-Bot\mediassist-frontend
npm run dev
```

Expected output:

```
Local:        http://localhost:5173/
```

---

## 📂 Directory Structure Created/Used

```
d:\AI Project\MediAssist-Bot\
├── .env.local ..................... ✅ Your configuration file (CREATE THIS)
├── data/
│   ├── vector_db.sqlite3 .......... ✅ Auto-created by RAG (vectorized drugs)
│   ├── conversations.db ........... ✅ Auto-created by persistence (chat history)
│   └── processed/
│       └── drug_knowledge_bot_ready_clean.csv
│
├── app/
│   ├── main.py .................... ✅ RAG endpoints & init
│   ├── services/
│   │   ├── rag_vector_search.py ... ✅ RAG core service
│   │   ├── llm_service.py ......... ✅ Enhanced with RAG
│   │   ├── conversation_memory.py . ✅ Persistence enabled
│   │   ├── drug_lookup.py
│   │   ├── chat_service.py
│   │   ├── medical_safety.py
│   │   └── ...
│   └── ...
│
├── tests/
│   ├── test_rag_service.py ........ ✅ RAG tests
│   └── ...
│
└── requirements-rag.txt ........... ✅ Optional (scikit-learn for TF-IDF)
```

---

## 🧪 Verify Your Setup

### 1. Check RAG Status

```bash
curl http://localhost:8000/api/rag/status
```

**Expected response:**

```json
{
  "rag_status": {
    "rag_enabled": true,
    "provider": "tfidf",
    "dimension": 100,
    "vector_count": 2847,
    "db_path": "./data/vector_db.sqlite3"
  }
}
```

### 2. Test Semantic Search

```bash
curl "http://localhost:8000/api/rag/search?query=diabetes+medication&top_k=3"
```

**Expected response:**

```json
{
  "query": "diabetes medication",
  "top_k": 3,
  "count": 3,
  "results": [
    {
      "drug_name": "Metformin",
      "similarity": 0.92,
      "metadata": { "indications": "Type 2 diabetes management" }
    }
  ]
}
```

### 3. Test Chat with RAG

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "diabetes",
    "age": 45,
    "message": "What drugs help diabetes?",
    "session_id": "test-1"
  }'
```

### 4. Check Conversation Persistence

```bash
# Verify database was created
ls -la d:\AI Project\MediAssist-Bot\data\
# Should show:
# - conversations.db (if USE_CONVERSATION_PERSISTENCE=1)
# - vector_db.sqlite3
```

---

## 📝 Configuration Reference

### Your TF-IDF Setup

```env
# Main RAG settings
RAG_ENABLED=1                          # Enable RAG
EMBEDDING_PROVIDER=tfidf               # Use TF-IDF (no deps needed)

# Search parameters
SIMILARITY_THRESHOLD=0.5               # Minimum match quality (0-1)
TOP_K_RESULTS=5                        # Return top 5 results

# Storage
VECTOR_DB_PATH=./data/vector_db.sqlite3  # Where vectors stored

# Persistence
USE_CONVERSATION_PERSISTENCE=1         # Save chats to database
```

### If You Switch Providers Later

```env
# Switch to OpenAI
EMBEDDING_PROVIDER=openai
LLM_API_KEY=sk-your-key-here

# Switch to Local (offline)
EMBEDDING_PROVIDER=local
# (requires: pip install sentence-transformers)

# Back to TF-IDF
EMBEDDING_PROVIDER=tfidf
# (no additional setup needed)
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: RAG shows as disabled

```
"rag_enabled": false
```

**Solution:**

- Check `.env.local` file exists
- Verify `RAG_ENABLED=1` (not 0)
- Restart backend with `uvicorn app.main:app --reload`

### Issue 2: Vector database not created

```
"vector_count": 0
```

**Solution:**

- Wait for backend to fully start (check logs for "RAG vectorization complete")
- Check file permissions in `data/` directory
- Ensure `scikit-learn` installed (for TF-IDF): `pip install scikit-learn`

### Issue 3: Conversation persistence not working

**Solution:**

- Check `USE_CONVERSATION_PERSISTENCE=1` is set
- Look for `conversations.db` in `data/` folder
- Verify write permissions to `data/` directory

### Issue 4: Slow first startup

**Normal behavior:**

- First run: 5-10 seconds (vectorizing 2800+ drugs)
- Subsequent runs: < 1 second (loading from database)

---

## 📊 File Summary

| File                                  | Type     | Purpose                | Status          |
| ------------------------------------- | -------- | ---------------------- | --------------- |
| `.env.local`                          | Config   | Your environment setup | ✅ Created      |
| `app/main.py`                         | Code     | RAG endpoints & init   | ✅ Updated      |
| `app/services/rag_vector_search.py`   | Code     | RAG implementation     | ✅ Existing     |
| `app/services/llm_service.py`         | Code     | LLM + RAG integration  | ✅ Updated      |
| `app/services/conversation_memory.py` | Code     | Chat persistence       | ✅ Existing     |
| `tests/test_rag_service.py`           | Tests    | RAG testing            | ✅ Existing     |
| `data/vector_db.sqlite3`              | Database | Vectorized drugs       | ✅ Auto-created |
| `data/conversations.db`               | Database | Chat history           | ✅ Auto-created |

---

## 🎯 Next Steps

1. ✅ Create `.env.local` (already done for you)
2. ✅ Set environment variables
3. ✅ Run: `uvicorn app.main:app --reload`
4. ✅ Verify status at `/api/rag/status`
5. ✅ Test search at `/api/rag/search?query=...`
6. ✅ Run tests: `pytest tests/ -v`

**You're all set! TF-IDF RAG is ready to use.** 🚀
