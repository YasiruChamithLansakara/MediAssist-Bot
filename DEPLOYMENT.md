# Deploying MediAssist

One container serves the API and the UI together. It is self-contained: the
drug dataset, the embedding model and the prebuilt retrieval index are all
baked into the image, so the running container needs no outbound network
except to the LLM provider.

---

## 1. Before you deploy

Two values are required. The container **refuses to start** without the first.

```bash
cp .env.docker.example .env.docker
python3 -c "import secrets; print(secrets.token_urlsafe(12))"   # your ACCESS_CODE
```

Fill in `.env.docker`:

| Variable | Why it matters |
|---|---|
| `ACCESS_CODE` | Gates every endpoint except `/api/health` and `/api/config`. Without it a medical-sounding tool sits open to crawlers and anyone can spend your LLM quota. Minimum 8 characters; 16+ is better. |
| `LLM_API_KEY` | Free Groq key from [console.groq.com](https://console.groq.com). Without it the app still runs, but answers drop to rule-based text. |

`.env.docker` is gitignored. Keep it that way.

---

## 2. Build and run

```bash
docker compose up -d --build
docker compose logs -f
```

The build takes roughly 10–15 minutes the first time — most of it downloading
PyTorch and building the retrieval index. Subsequent builds reuse the layers.

Check it came up:

```bash
curl -s localhost:8000/api/health
# {"status":"ok"}

curl -s localhost:8000/api/config
# {"access_required":true,"access_header":"X-Access-Code"}

curl -s -H "X-Access-Code: YOUR_CODE" localhost:8000/api/meta | head -c 300
```

Then open `http://localhost:8000` and enter the code.

---

## 3. Put TLS in front of it

The compose file binds the container to `127.0.0.1:8000` on purpose — it is
**not** reachable from the internet as shipped. Terminate TLS with a reverse
proxy before exposing it. Serving a medication tool over plain HTTP means the
access code travels in clear text.

**Caddy** (simplest — certificates are automatic):

```caddyfile
mediassist.example.com {
    reverse_proxy 127.0.0.1:8000
    encode gzip
    request_body {
        max_size 6MB          # prescription uploads are capped at 5 MB
    }
}
```

**nginx**:

```nginx
server {
    listen 443 ssl http2;
    server_name mediassist.example.com;

    ssl_certificate     /etc/letsencrypt/live/mediassist.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mediassist.example.com/privkey.pem;

    client_max_body_size 6M;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # OCR on a large photo can take a while on a small VPS.
        proxy_read_timeout 180s;
    }
}
```

The container already runs uvicorn with `--proxy-headers`, so client IPs in
the logs and in the rate limiter are the real ones rather than the proxy's.

---

## 4. Sizing

| Resource | Needed | Why |
|---|---|---|
| RAM | **3 GB** | torch, faiss and the easyocr model total roughly 1.5 GB resident; the limit leaves headroom for OCR on a large image. |
| Disk | ~6 GB | The image carries torch, the embedding model and the prebuilt index. |
| CPU | 2 cores | Retrieval is milliseconds; OCR is the slow part and is CPU-bound. |

A 4 GB VPS is comfortable. A 1 GB instance will be OOM-killed during OCR.

**Do not raise `--workers`.** The rate limiter, conversation memory and
retrieval index are per-process singletons: a second worker gives users
inconsistent chat history and doubles the effective rate limit. To scale, run
more containers behind a load balancer with sticky sessions — or move that
state into Redis first.

---

## 5. What the deployment does with data

This matters for a health-adjacent tool, and the defaults are deliberate.

* **Chat history stays in memory.** `USE_CONVERSATION_PERSISTENCE=0` means
  nothing is written to disk. Sessions expire after `MEMORY_TTL_HOURS` (2 by
  default) and at most `MEMORY_MAX_SESSIONS` are held at once.
* **Only a summary is retained per turn** — the message text, the drug names
  matched and the intent. Full label text is not copied into the session.
* **Uploaded images are never stored.** They are processed in memory and
  discarded when the request ends.
* **Users can erase their own conversation** with the Clear button, which
  calls `POST /api/chat/forget`.

If you turn persistence on, the app logs a warning telling you what you have
just taken on: encrypt the volume, set a retention policy, and publish a
privacy notice.

---

## 6. Operating it

**Is the AI actually working?** The single most useful check — this is exactly
what was missing when the previous LLM was retired and the app degraded to
rule-based answers silently for three weeks:

```bash
curl -s -H "X-Access-Code: YOUR_CODE" localhost:8000/api/meta \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['llm'])"
```

`available: true` with a `model` name and `last_error: null` is healthy. The
UI shows the same thing in the System panel.

**Rotating the access code:** edit `.env.docker`, then
`docker compose up -d`. Existing users are prompted for the new code on their
next request; nothing else is invalidated.

**Updating the drug dataset:** rebuild the image. The retrieval index is
stamped with the dataset path and rebuilds itself when the data changes, so a
stale index cannot survive a dataset swap.

**Logs** are JSON-file capped at 3 × 10 MB. Every request carries an
`X-Request-ID` that appears in the log line and in error responses, so a user
report maps to a specific request.

---

## 7. Before sharing the link

- [ ] `ACCESS_CODE` set to something you did not reuse elsewhere
- [ ] HTTPS working, HTTP redirecting to it
- [ ] `curl` to `/api/meta` without the code returns **401**
- [ ] `llm.available` is `true`
- [ ] A Groq spend limit set on the account, in case the code leaks
- [ ] You have read the disclaimer the app shows and are comfortable that the
      people you share it with will understand this is an educational tool,
      not medical advice

---

## Environment reference

| Variable | Default | Notes |
|---|---|---|
| `ACCESS_CODE` | — | **Required in production.** Startup fails without it. |
| `LLM_API_KEY` | — | Groq key. Absent → rule-based answers. |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Retired ids are rejected at startup. |
| `LLM_MODEL_CHAIN` | 3 models | Tried in order if the active one is rejected. |
| `ENV` | `production` | `development` disables the access gate. |
| `RATE_LIMIT_RPM` | `60` | Per IP, per minute. |
| `MEMORY_TTL_HOURS` | `2` | How long a conversation survives. |
| `MEMORY_MAX_SESSIONS` | `500` | Ceiling before least-recently-used eviction. |
| `USE_CONVERSATION_PERSISTENCE` | `0` | `1` writes chat turns to SQLite — read §5 first. |
| `MAX_UPLOAD_BYTES` | `5242880` | Prescription image cap. |
| `ALLOWED_ORIGINS` | empty | Only needed if the UI is served from another origin. |
| `FRONTEND_DIST` | bundled | Path to the built UI; the image sets this up already. |
