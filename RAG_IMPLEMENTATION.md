# RAG Vector Search Implementation Guide

## Overview

**RAG (Retrieval-Augmented Generation)** enhances MediAssist's AI capabilities by enabling semantic search over the drug knowledge base. This allows the LLM to retrieve contextually relevant drug information beyond keyword matching.

## Features

✅ **Multiple Embedding Backends**

- OpenAI embeddings (highest quality, costs money)
- Local sentence-transformers (offline, free, decent quality)
- TF-IDF fallback (lightweight, for testing)

✅ **Efficient Vector Storage**

- SQLite-based vector database
- Automatic drug knowledge vectorization on startup
- Cosine similarity search

✅ **Seamless LLM Integration**

- RAG automatically enhances LLM context
- Falls back gracefully if unavailable
- No changes needed to existing chat API

✅ **Production Ready**

- Thread-safe operations
- Comprehensive error handling
- Configurable similarity threshold and result limits

## Quick Start

### 1. Basic Setup (TF-IDF - Testing)

TF-IDF requires no external dependencies:

```bash
# TF-IDF is built-in to requirements.txt
# Just set environment variable
export RAG_ENABLED=1
export EMBEDDING_PROVIDER=tfidf

# Run backend
uvicorn app.main:app --reload
```

RAG starts automatically on backend startup and vectorizes the drug knowledge base.

### 2. Production Setup (OpenAI)

For production with highest quality embeddings:

```bash
# Install OpenAI (you likely have this already for LLM)
pip install openai

# Set environment variables
export RAG_ENABLED=1
export EMBEDDING_PROVIDER=openai
export LLM_API_KEY=sk-...  # Uses same key as LLM

# Run backend
uvicorn app.main:app --reload
```

The OpenAI embeddings use the `text-embedding-3-small` model (1536 dimensions), same provider as your LLM.

### 3. Privacy Option (Local Embeddings)

For offline operation with no API calls:

```bash
# Install sentence-transformers
pip install -r requirements-rag.txt

# Set environment variables
export RAG_ENABLED=1
export EMBEDDING_PROVIDER=local
export LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Run backend
uvicorn app.main:app --reload
```

The first run downloads the model (~80MB). Subsequent runs are instant.

## Environment Variables

### Core Configuration

```env
# Enable/disable RAG
RAG_ENABLED=1  # or 0 to disable

# Choose embedding provider: openai, local, or tfidf
EMBEDDING_PROVIDER=openai
```

### OpenAI Embeddings

```env
# Uses LLM_API_KEY automatically
EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-3-large (3072 dims)
EMBEDDING_DIMENSION=1536  # text-embedding-3-small
```

### Local Embeddings

```env
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Other options:
# - sentence-transformers/all-MiniLM-L6-v2 (384 dims, fastest)
# - sentence-transformers/all-mpnet-base-v2 (768 dims, balanced)
# - sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2 (384 dims, multilingual)
```

### Search Parameters

```env
SIMILARITY_THRESHOLD=0.5  # Min similarity score (0-1)
TOP_K_RESULTS=5  # Number of results to return

VECTOR_DB_PATH=./data/vector_db.sqlite3  # Storage location
```

## API Endpoints

### Semantic Search

```bash
# Search for semantically similar drugs
GET /api/rag/search?query=diabetes+medication&top_k=5

# Response
{
  "query": "diabetes medication",
  "top_k": 5,
  "count": 3,
  "results": [
    {
      "drug_name": "Metformin",
      "similarity": 0.92,
      "metadata": {
        "indications": "Type 2 diabetes management",
        "warnings": "Lactic acidosis risk"
      }
    },
    ...
  ]
}
```

### RAG Status

```bash
# Check RAG availability and database stats
GET /api/rag/status

# Response
{
  "rag_status": {
    "rag_enabled": true,
    "provider": "openai",
    "dimension": 1536,
    "vector_count": 2847,
    "db_path": "./data/vector_db.sqlite3"
  }
}
```

### Memory Stats (includes RAG)

```bash
# Get memory and RAG statistics
GET /api/memory/stats

# Response
{
  "stats": {
    "active_sessions": 2,
    "total_turns": 47
  },
  "llm_available": true,
  "rag_status": {
    "rag_enabled": true,
    "provider": "openai",
    "vector_count": 2847
  }
}
```

## How RAG Enhances Chat

### Without RAG

```
User: "What helps diabetes?"
↓
Chat Service: Matches "diabetes" against drug database
↓
LLM: Responds based on direct matches only
↓
Response: Limited to drugs with "diabetes" keyword
```

### With RAG

```
User: "What helps diabetes?"
↓
Chat Service: Matches "diabetes" + RAG retrieves semantic matches
↓
RAG: Searches vector DB → finds Metformin, Insulin, GLP-1 agonists
↓
LLM: Receives both direct matches + RAG results
↓
Response: Rich, comprehensive answer grounded in actual data
```

## Python Integration

### Retrieve Context

```python
from app.services.rag_vector_search import retrieve_drug_context

# Get semantic matches for a query
results = retrieve_drug_context("diabetes management", top_k=5)

for result in results:
    print(f"{result['drug_name']}: {result['similarity']:.2%} match")
    print(f"  Notes: {result['metadata'].get('indications', '')}")
```

### Vectorize Drug Knowledge

```python
from app.services.rag_vector_search import vectorize_drug_knowledge

drugs = [
    {
        "name": "Metformin",
        "generic_name": "metformin",
        "sections": {
            "indications": "Type 2 diabetes",
            "warnings": "Lactic acidosis",
            "contraindications": "Kidney disease"
        },
        "disease": "diabetes"
    },
    # ... more drugs
]

count = vectorize_drug_knowledge(drugs)
print(f"Vectorized {count} drugs")
```

### Get RAG Service Status

```python
from app.services.llm_service import get_rag_status

status = get_rag_status()
print(f"RAG enabled: {status['rag_enabled']}")
print(f"Provider: {status['provider']}")
print(f"Vectors in DB: {status['vector_count']}")
```

## Troubleshooting

### RAG Not Working

**Check status:**

```bash
curl http://localhost:8000/api/rag/status
```

**If `rag_enabled: false`:**

1. Check environment variable:

   ```bash
   echo $RAG_ENABLED  # Should be 1
   ```

2. Check logs for warnings about missing dependencies:

   ```
   WARNING | sentence-transformers not installed
   WARNING | No LLM_API_KEY for OpenAI embeddings
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements-rag.txt
   ```

### Slow Search Response

RAG search may be slow if:

- **First startup**: Vectorizing all drugs (~5-10 seconds)
- **Large database**: 10k+ drugs
- **Local model**: Slower inference than OpenAI

Solutions:

- Reduce `TOP_K_RESULTS` (default 5)
- Increase `SIMILARITY_THRESHOLD` to filter noise (default 0.5)
- Use OpenAI for faster responses

### Out of Memory

If running on limited hardware:

- Use TF-IDF instead of local embeddings
- Reduce number of drugs vectorized
- Use sentence-transformers/all-MiniLM-L6-v2 (smallest model)

### Search Returns No Results

1. Check if database has vectors:

   ```bash
   curl http://localhost:8000/api/rag/status
   # Should show vector_count > 0
   ```

2. Lower `SIMILARITY_THRESHOLD`:

   ```env
   SIMILARITY_THRESHOLD=0.3  # More permissive
   ```

3. Increase `TOP_K_RESULTS`:
   ```env
   TOP_K_RESULTS=10
   ```

## Performance Characteristics

### Embedding Generation

| Provider | Speed      | Quality | Cost         | Privacy |
| -------- | ---------- | ------- | ------------ | ------- |
| OpenAI   | 1-2s/batch | Highest | $0.02 per 1M | Via API |
| Local    | 0.5-2s     | Good    | Free         | Local   |
| TF-IDF   | <100ms     | Fair    | Free         | Local   |

### Search Speed

- **Cold start**: ~100-500ms (model loading)
- **Subsequent**: ~10-50ms (in-memory DB)
- **Large DB**: ~50-200ms (1000+ vectors)

### Storage

- **Vector DB size**: ~1-2MB per 1000 drugs
- **Model size**:
  - OpenAI: API-only (no local storage)
  - sentence-transformers: 80-700MB depending on model
  - TF-IDF: None (generated on-the-fly)

## Production Deployment

### Railway

Set environment variables in Railway dashboard:

```env
RAG_ENABLED=1
EMBEDDING_PROVIDER=openai
# LLM_API_KEY already set for LLM
```

On startup:

- Drug lookup store initializes
- RAG service starts with OpenAI embeddings
- All 2000+ drugs automatically vectorized
- Search endpoint ready for queries

### Multiple Instances

For multi-instance deployment:

- Each instance has its own vector DB (in-memory + SQLite)
- Consistent results across instances
- No synchronization needed (read-only after startup)

### Monitoring

Check RAG health:

```bash
curl https://your-app.railway.app/api/rag/status

# Expected response
{
  "rag_status": {
    "rag_enabled": true,
    "provider": "openai",
    "vector_count": 2847,
    "dimension": 1536
  }
}
```

## Examples

### Medical Student Query

```python
# Student asks: "What medications treat hypertension?"
query = "medications for hypertension management"
results = retrieve_drug_context(query, top_k=3)

# Returns semantic matches:
# - Lisinopril (ACE inhibitor)
# - Metoprolol (Beta blocker)
# - Hydrochlorothiazide (Diuretic)
```

### Complex Patient Case

```python
# Patient with diabetes and hypertension
# LLM receives:
# 1. Direct matches for diabetes drugs (keyword-based)
# 2. RAG matches for hypertension management
# 3. Contraindication warnings
# → Comprehensive, contextualized response
```

### Rare Drug Query

```python
# User asks about "dabigatran" (rare anticoagulant)
# Keyword search might miss similar drugs
# RAG finds semantic matches:
# - Rivaroxaban (similar mechanism)
# - Warfarin (similar indication)
# - Aspirin (related use case)
```

## Future Enhancements

- ✨ Hybrid BM25 + vector search for better recall
- ✨ FAISS integration for million+ drug databases
- ✨ Reranking models to improve top-k results
- ✨ Drug interaction vector clustering
- ✨ Side effect semantic similarity
- ✨ Multi-language embeddings for international drugs

## Testing

```bash
# Run RAG tests
pytest tests/test_rag_service.py -v

# Run full test suite including RAG
pytest tests/ -v
```

## References

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Sentence-Transformers](https://www.sbert.net/)
- [RAG Papers & Research](https://arxiv.org/abs/2005.11401)
- [Vector Database Design](https://www.pinecone.io/learn/vector-database/)
