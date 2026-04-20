"""
FastAPI server exposing the portfolio tool-calling agent.

POST /chat       { "question": str, "thread_id"?: str }
                 -> { "answer", "scroll_to", "component", "source", "thread_id" }

GET  /health     -> { "status": "ok" }
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import run as run_agent

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
    component: Optional[dict] = None
    source: str = "profile"
    thread_id: str

@api.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
        component=out.get("component"),
        source=out.get("source") or "profile",
        thread_id=thread_id,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000, reload=False)
