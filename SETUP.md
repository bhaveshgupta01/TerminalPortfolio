# Portfolio — local setup

Two processes: **Vite frontend** and **FastAPI agent backend**.

## 1. Backend (agent)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# paste your key into .env:   GOOGLE_API_KEY=...

python ingest.py --reset      # builds ./chroma_db from knowledge/
python server.py              # FastAPI on http://localhost:8000
```

Health check: `curl http://localhost:8000/health` → `{"status":"ok"}`

Chat smoke test:
```bash
curl -sX POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"what did you build at TCS?"}' | jq
```

## 2. Frontend (Vite)

```bash
# from project root
cp .env.local.example .env.local   # optional — only needed to override API URL
npm install
npm run dev                         # http://localhost:5173
```

## 3. Deploy (later)

- Frontend → Vercel (`libralpanda.vercel.app`). Set `VITE_AGENT_API` env var to the backend URL.
- Backend → Railway / Render / Cloud Run (needs persistent disk for ChromaDB or swap to managed vector store).

## Architecture

```
Browser (React)
    │   POST /chat { question, thread_id }
    ▼
FastAPI (backend/server.py)
    ▼
LangGraph agent (backend/agent.py)
    ├─ classify         (Gemini: is this about Bhavesh?)
    ├─ retrieve         (ChromaDB over knowledge/)       ── about-me path
    ├─ generate_profile (Gemini, first-person JSON)
    ├─ web_search       (DuckDuckGo)                      ── general path
    └─ generate_web     (Gemini, first-person JSON)
    ▼
{ answer, scroll_to, component, source }
```

## Knowledge updates

Edit files in `backend/knowledge/`, then rerun `python ingest.py --reset`.
