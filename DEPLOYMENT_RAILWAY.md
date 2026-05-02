# Railway Deployment Guide for MediAssist Bot

## Overview

MediAssist Bot is designed to deploy seamlessly to Railway. This guide covers all deployment steps, including optional LLM features, database setup, and environment configuration.

## Prerequisites

- Railway account (free tier available at railway.app)
- GitHub repository with MediAssist Bot code
- (Optional) OpenAI API key for LLM features

## Deployment Architecture

```text
┌─────────────┐
│  Frontend   │  (Static files + Vite)
│  (React)    │
└──────┬──────┘
       │ VITE_API_BASE
       ▼
┌─────────────┐        ┌──────────────┐
│  Backend    │◄──────►│   Database   │
│  (FastAPI)  │        │  (SQLite)    │
└─────────────┘        └──────────────┘
       │
       ├─► Tesseract (OCR)
       ├─► spaCy (NER)
       └─► OpenAI API (Optional)
```

## Step 1: Prepare Repository

### 1.1 Ensure Git is set up

```bash
git init
git add .
git commit -m "Initial commit for Railway deployment"
git push origin main
```

### 1.2 Verify required files exist

- `railway.toml` - Railway config (RAILPACK builder)
- `railpack.json` - Package dependencies (Python 3.12 + Tesseract)
- `requirements.txt` - Core Python dependencies
- `app/main.py` - FastAPI application entry point
- `mediassist-frontend/` - Frontend code

## Step 2: Create Railway Project

### 2.1 Via Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Create new project
railway init
```

### 2.2 Via Railway Dashboard

1. Go to [Railway](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authorize and select MediAssist-Bot repository
5. Select branch (e.g., main)

## Step 3: Configure Environment Variables

Set these in Railway dashboard under "Variables":

### Core Variables (Required)

```env
ENV=production
LOG_LEVEL=INFO

# CORS and API configuration
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://your-frontend-railway-domain.railway.app
RATE_LIMIT_ENABLED=1
RATE_LIMIT_RPM=120
```

### Optional: LLM Features (Advanced Chat)

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...your-openai-key...
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
```

### Optional: Conversation Persistence

```env
USE_CONVERSATION_PERSISTENCE=1
CONVERSATION_DB=/tmp/conversations.db
```

### Optional: NER Enhancement

```env
# spaCy will auto-download model on first use
# Requires ~400MB disk space
```

## Step 4: Deploy Backend

### 4.1 Initial Deployment

```bash
railway up
```

Or via dashboard:

1. Go to your Railway project
2. Click "Deploy"
3. Select branch and click "Deploy"

### 4.2 Monitor Deployment

```bash
railway logs -f
```

Watch for:

- ✅ Server started on 0.0.0.0:PORT
- ✅ /api/health returns 200
- ⚠️ Tesseract available: True (or False with fallback)
- ⚠️ spaCy available: True (or False with fallback)

### 4.3 Get Backend URL

```bash
railway open
```

Your backend URL will be something like:

```text
https://mediassist-bot-prod.railway.app
```

## Step 5: Deploy Frontend (Optional)

### 5.1 Build Frontend

```bash
cd mediassist-frontend
npm install
npm run build
```

### 5.2 Deploy Frontend Separately (Recommended)

#### Option A: Railway Static Service

1. Create new Railway service for static files
2. Configure to serve `dist/` folder
3. Add as separate service in Railway

#### Option B: Netlify/Vercel

1. Build: `npm run build`
2. Deploy `dist/` folder to Netlify or Vercel
3. Configure environment variable:

   ```env
   VITE_API_BASE=https://your-railway-backend.railway.app/api
   ```

#### Option C: Same Railway Container

1. Configure railway.toml to serve static files
2. Add build step: `npm run build` in frontend folder
3. Include frontend files in production build

## Step 6: Configure CORS and Frontend URL

After frontend is deployed, update backend environment:

```bash
railway variables set ALLOWED_ORIGINS=https://your-frontend-url.com
```

Or via dashboard Variables:

```env
ALLOWED_ORIGINS=https://your-frontend-railway.railway.app,https://your-frontend-vercel.app
```

## Step 7: Verify Deployment

### 7.1 Test Backend

```bash
# From your local machine
curl https://your-railway-backend.railway.app/api/health
# Should return: {"status":"ok"}

curl https://your-railway-backend.railway.app/api/meta
# Should return metadata with supported diseases
```

### 7.2 Test OCR

```bash
# Test that Tesseract is available
curl https://your-railway-backend.railway.app/api/meta | grep ocr_runtime
```

### 7.3 Test Chat (without LLM)

```bash
curl -X POST https://your-railway-backend.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "diabetes",
    "age": 45,
    "message": "Is metformin safe?"
  }'
```

### 7.4 Test Chat with LLM (Optional)

If LLM_API_KEY is configured:

```bash
# Should include "answer_source": "llm_grounded"
```

## Step 8: Optional - Add Persistence

### 8.1 Database (SQLite)

Conversation memory uses SQLite by default:

- File: `conversations.db` (auto-created)
- Persists when `USE_CONVERSATION_PERSISTENCE=1`
- In Railway, stored in container `/tmp/` (non-persistent between restarts)

### 8.2 Add PostgreSQL (Production)

```bash
railway add
# Select PostgreSQL
railway variables set DATABASE_URL=... (auto-set by Railway)
```

Then update conversation_memory.py to use PostgreSQL.

### 8.3 Add Redis (Optional - Caching)

```bash
railway add
# Select Redis
```

## Step 9: Monitoring & Logs

### Monitor via Railway Dashboard

- Go to your project
- Click "Deployments" tab
- View real-time logs

### View Logs via CLI

```bash
railway logs -f
```

### Metrics to Watch

- CPU usage
- Memory usage
- Request latency
- Error rates

## Troubleshooting

### Issue: "Module not found: spacy"

**Solution:** spaCy is optional. NER will fall back to rule-based system.

### Issue: "LLM is returning None"

**Solution:** Check `LLM_API_KEY` is set and valid. View logs for detailed errors.

### Issue: OCR not working

**Solution:** Check logs for "Tesseract" status. May require system package installation.

### Issue: CORS errors

**Solution:** Update `ALLOWED_ORIGINS` to include frontend domain.

## Advanced Configuration

### Enable Verbose Logging

```bash
railway variables set LOG_LEVEL=DEBUG
```

### Rate Limiting

```bash
railway variables set RATE_LIMIT_RPM=60  # Reduce from 120
```

### Frontend Env (for frontend build)

```env
# .env.production
VITE_API_BASE=https://your-railway-backend.railway.app/api
```

## Production Checklist

- [ ] Backend deployed and health check passing
- [ ] `ALLOWED_ORIGINS` configured for frontend domain
- [ ] Frontend deployed and accessing API
- [ ] CORS requests working
- [ ] Rate limiting enabled
- [ ] Logging level set to INFO (not DEBUG)
- [ ] LLM features tested (if using OpenAI)
- [ ] OCR tested with sample prescription image
- [ ] Conversation history working (check `/api/chat/history/{session_id}`)
- [ ] Error handling and fallbacks verified
- [ ] Security: API keys not in logs
- [ ] Security: Medical disclaimers displayed

## Performance Tuning

### For High Traffic

```bash
railway variables set RATE_LIMIT_BURST=50  # Increase burst
railway variables set LLM_MAX_TOKENS=512   # Reduce for faster responses
```

### For Lower Costs

```bash
railway variables set LLM_PROVIDER=  # Disable LLM
railway variables set USE_CONVERSATION_PERSISTENCE=0  # Disable persistence
```

## Rollback

If deployment has issues:

```bash
railway rollback
# Select previous deployment
```

## Next Steps

1. **Monitor metrics** - Watch CPU, memory, response times
2. **Collect feedback** - Test with real users
3. **Scale if needed** - Upgrade Railway plan if hitting limits
4. **Add monitoring** - Integrate with Sentry, DataDog, etc.
5. **Implement RAG** - Add vector search for better responses
6. **Custom domain** - Add custom domain to Railway project

## Support

- Railway docs: [Railway Docs](https://docs.railway.app)
- Railway support: [Railway Support](https://railway.app/support)
- MediAssist issues: [MediAssist Issues](https://github.com/your-repo/issues)

---

**Last updated:** April 26, 2026
**Version:** 1.0
