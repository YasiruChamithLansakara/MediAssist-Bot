# MediAssist-Bot - Quick Start Guide

## 🚀 Local Development (5 minutes)

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

### 3. Start Frontend (in new terminal)

```bash
cd mediassist-frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

### 4. Test Features

- **Lookup Tab**: Search for any drug name (aspirin, metformin, etc.)
- **Chat Tab**: Ask questions about medications
- **Prescription Tab**: Upload a prescription image for OCR

---

## 🔧 Optional: Enable Advanced Features

### Enable LLM (ChatGPT Integration)

```bash
pip install -r requirements-llm.txt
export LLM_API_KEY=sk-YOUR_OPENAI_KEY
```

### Enable RAG Vector Search (Enhanced Context Retrieval)

```bash
pip install -r requirements-rag.txt
# RAG automatically initializes on startup
# Query at: POST /api/rag/search?query=your_query&top_k=5
```

**RAG Providers (choose one):**

- **OpenAI** (recommended): Uses same API key as LLM, semantic search
  - Set: `EMBEDDING_PROVIDER=openai`
- **Local** (offline): sentence-transformers, no external API calls
  - Set: `EMBEDDING_PROVIDER=local`
- **TF-IDF** (fallback): Works without extra deps, useful for testing
  - Set: `EMBEDDING_PROVIDER=tfidf`

### Enable spaCy Medical NER

```bash
pip install spacy
python -m spacy download en_core_sci_sm
```

### Enable Conversation Persistence

```bash
export USE_CONVERSATION_PERSISTENCE=1
```

---

## ✅ Run Tests

```bash
pytest tests/ -v
# Expected: 49 passed
```

---

## 🚢 Deploy to Railway (10 minutes)

1. **Push to GitHub**

   ```bash
   git add .
   git commit -m "MediAssist-Bot ready for deployment"
   git push
   ```

2. **Create Railway Project**
   - Visit https://railway.app
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository

3. **Set Environment Variables**
   - In Railway dashboard, go to "Variables"
   - Add: `ALLOWED_ORIGINS=*`
   - Optional: Add `LLM_API_KEY=sk-...` for ChatGPT

4. **Deploy**
   - Railway auto-deploys on push
   - Check deployment at: `https://your-app.railway.app`

---

## 📋 API Examples

### Health Check

```bash
curl http://localhost:8000/api/health
```

### Drug Lookup

```bash
curl -X POST http://localhost:8000/api/drug-lookup \
  -H "Content-Type: application/json" \
  -d '{"drug_name": "aspirin"}'
```

### Chat with Context

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is aspirin used for?",
    "session_id": "user-123"
  }'
```

### Get Chat History

```bash
curl http://localhost:8000/api/chat/history/user-123
```

### RAG Vector Search

```bash
# Search for semantically similar drugs (if RAG enabled)
curl "http://localhost:8000/api/rag/search?query=diabetes%20medication&top_k=5"
```

### Get RAG Status

```bash
curl http://localhost:8000/api/rag/status
```

---

## 🐛 Troubleshooting

### Backend won't start

- Check port 8000 is free: `netstat -an | find "8000"`
- Check Python 3.12+: `python --version`

### Frontend won't start

- Clear node_modules: `rm -r mediassist-frontend/node_modules && npm install`
- Check Node 18+: `node --version`

### OCR not working

- Tesseract must be installed (Railway installs via railpack automatically)
- Local testing: Install Tesseract OCR separately

### Tests failing

- Install all requirements: `pip install -r requirements.txt pytest`
- Run: `pytest tests/ -v`

---

## 📚 Documentation Files

- [DEPLOYMENT_RAILWAY.md](DEPLOYMENT_RAILWAY.md) - Complete Railway deployment guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - All features explained
- [VALIDATION_COMPLETE.md](VALIDATION_COMPLETE.md) - Test results & validation

---

## 🎯 Next Steps

1. Run tests locally: `pytest tests/ -v` → Should see **49 passed**
2. Start dev servers (backend + frontend)
3. Test one feature (e.g., drug lookup for "aspirin")
4. Deploy to Railway following DEPLOYMENT_RAILWAY.md
5. Share production URL!

---

**Status**: ✅ Ready to use locally or deploy!
