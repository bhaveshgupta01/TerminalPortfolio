# Portfolio Agent (Gemini + LangGraph + ChromaDB)

First-person AI twin of Bhavesh, grounded in the curated `knowledge/` corpus.
Falls back to DuckDuckGo web search for questions that aren't about Bhavesh.

## Response shape

```json
{
  "answer": "first-person reply",
  "scroll_to": "experience | skills | projects | ... | null",
  "component": null | { "type": "stats|timeline|cards|comparison|list|code|quote", "title": "...", "data": {...} },
  "source": "profile" | "web"
}
```

## Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste GOOGLE_API_KEY
python ingest.py --reset      # build the vector DB
python server.py              # FastAPI on :8000
```

## Test

```bash
curl -sX POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question":"what did you build at TCS?"}' | jq
```

## Graph

```
classify -> retrieve -> generate_profile -> END   (about Bhavesh)
classify -> web_search -> generate_web -> END     (general)
```
