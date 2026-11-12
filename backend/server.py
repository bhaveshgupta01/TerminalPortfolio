"""
FastAPI server exposing the portfolio tool-calling agent.

POST /chat         { "question": str, "thread_id"?: str }
                   -> { "answer", "scroll_to", "focus", "component", "source", "thread_id" }

POST /chat/stream  { "question": str, "thread_id"?: str }
                   -> text/event-stream (SSE) of:
                        data: {"type":"thread","thread_id":"..."}
                        data: {"type":"status","stage":"..."}
                        data: {"type":"tool_call",...}
                        data: {"type":"ui_command","action":"SET_ACTIVE_CARD",...}
                        data: {"type":"text_delta","content":"..."}
                        data: {"type":"component","data":{...}}
                        data: {"type":"final",...}
                        data: [DONE]

GET  /health       -> { "status": "ok" }
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import run as run_agent, run_stream as run_agent_stream

load_dotenv()

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:4173,https://libralpanda.vercel.app",
    ).split(",") if o.strip()
]

# Optional secret to gate the manual re-ingest endpoint.
INGEST_SECRET = os.getenv("INGEST_SECRET", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On boot: auto-populate ChromaDB if empty (no manual shell needed)."""
    try:
        from ingest import ensure_ingested
        ensure_ingested()
    except Exception as e:
        print(f"[startup] auto-ingest failed (server will still start): {e}")
    yield


api = FastAPI(title="Bhavesh Portfolio Agent", version="2.1.0", lifespan=lifespan)

api.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    scroll_to: Optional[str] = None
    focus: Optional[dict] = None
    component: Optional[dict] = None
    source: str = "profile"
    thread_id: str

@api.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api.get("/now")
def now() -> dict:
    """Content for the v2 'Now' card.

    Read from backend/now.json on every request so I can update what I'm
    working on without redeploying the agent. Tiny payload, cheap to serve.
    """
    path = os.path.join(os.path.dirname(__file__), "now.json")
    if not os.path.exists(path):
        return {"updated": "soon", "lines": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        return {"updated": "could not load", "lines": [], "error": str(e)}


@api.post("/admin/ingest")
def admin_ingest(x_ingest_secret: str = Header(default="")) -> dict:
    """Manual re-ingest endpoint — useful when knowledge/ changes without a
    full redeploy. Protected by the INGEST_SECRET env var; returns 401 if
    missing/wrong."""
    if not INGEST_SECRET or x_ingest_secret != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        from ingest import main as run_ingest
        run_ingest(reset=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")
    return {"status": "reingested"}

@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or f"anon-{uuid.uuid4().hex[:8]}"
    try:
        out = run_agent(req.question, thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent error: {e}")

    return ChatResponse(
        answer=(out.get("answer") or "").strip() or "I'm not sure how to answer that yet.",
        scroll_to=out.get("scroll_to"),
        focus=out.get("focus"),
        component=out.get("component"),
        source=out.get("source") or "profile",
        thread_id=thread_id,
    )


@api.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events stream for the lab UI.

    Emits discrete events (status, tool_call, ui_command, text_delta,
    component, final, error) so the frontend can choreograph the chat text
    and wallet-card transitions in lockstep with the agent's reasoning.
    """
    thread_id = req.thread_id or f"anon-{uuid.uuid4().hex[:8]}"

    async def event_source():
        def pack(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        yield pack({"type": "thread", "thread_id": thread_id})
        try:
            async for event in run_agent_stream(req.question, thread_id):
                yield pack(event)
        except Exception as e:  # pragma: no cover
            yield pack({"type": "error", "message": f"stream crashed: {e}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/Railway)
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000, reload=False)
