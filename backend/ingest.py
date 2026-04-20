"""
Ingest the curated knowledge/ folder into ChromaDB.

Each markdown file may start with an HTML comment declaring its role, e.g.:
    <!-- role: tcs -->
Known roles: personal, nyuad, tcs, synopsys, esec, gdsc, skills, projects,
             leadership, metrics, publication. Chunks missing a role are
             tagged as "general".

Run:   python ingest.py          # adds new files, keeps existing
       python ingest.py --reset  # wipes and rebuilds
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KNOWLEDGE_PATH = Path(os.getenv("KNOWLEDGE_PATH", "./knowledge"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./chroma_db"))
COLLECTION_NAME = "portfolio-profile"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

TEXT_EXTS = {".md", ".txt", ".json", ".py"}

KNOWN_ROLES = {
    "personal", "nyuad", "tcs", "synopsys", "esec", "gdsc",
    "skills", "projects", "leadership", "metrics", "publication",
    "general",
}

ROLE_MARKER = re.compile(r"<!--\s*role:\s*([a-z_]+)\s*-->", re.IGNORECASE)


def parse_role(text: str) -> str:
    match = ROLE_MARKER.search(text[:500])
    if match:
        role = match.group(1).lower()
        if role in KNOWN_ROLES:
            return role
        print(f"   ⚠ unknown role '{role}', falling back to 'general'")
    return "general"


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(str(path)).load()
        if ext == ".docx":
            return Docx2txtLoader(str(path)).load()
        if ext == ".csv":
            return CSVLoader(str(path)).load()
        if ext in TEXT_EXTS:
            return TextLoader(str(path), encoding="utf-8").load()
    except Exception as e:
        print(f"   ✗ failed to load {path.name}: {e}")
    return []


def get_existing_sources(vs: Chroma) -> set[str]:
    try:
        data = vs._collection.get(include=["metadatas"])
        return {m.get("source") for m in data.get("metadatas", []) if m and m.get("source")}
    except Exception:
        return set()


def main(reset: bool) -> None:
    if reset and CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print(f"✨ wiped {CHROMA_PATH}")

    if not KNOWLEDGE_PATH.exists():
        raise SystemExit(f"knowledge folder not found: {KNOWLEDGE_PATH}")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    existing = get_existing_sources(vs)
    print(f"📦 existing sources in DB: {len(existing)}")

    new_docs: list[Document] = []
    role_counts: dict[str, int] = {}

    for path in sorted(KNOWLEDGE_PATH.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        source = str(path)
        if source in existing:
            continue
        docs = load_file(path)
        if not docs:
            continue

        # one role per file (based on marker or default)
        role = parse_role(docs[0].page_content) if docs else "general"
        role_counts[role] = role_counts.get(role, 0) + 1

        for d in docs:
            d.metadata["source"] = source
            d.metadata["filename"] = path.name
            d.metadata["role"] = role

        print(f"   + {path.name} [role={role}] ({len(docs)} docs)")
        new_docs.extend(docs)

    if not new_docs:
        print("✅ nothing new to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = splitter.split_documents(new_docs)
    # preserve role on split chunks
    for s in splits:
        s.metadata.setdefault("role", "general")

    vs.add_documents(splits)
    print(f"🎉 added {len(splits)} chunks from {len(new_docs)} docs")
    print(f"   by role: {role_counts}")


def ensure_ingested() -> int:
    """Idempotent ingest — populates the DB only if empty.

    Safe to call on server startup: returns fast if the collection already has
    chunks, runs the full ingest otherwise. Returns the chunk count.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    try:
        count = vs._collection.count()
    except Exception:
        count = 0
    if count > 0:
        print(f"[ingest] ChromaDB already has {count} chunks — skipping.")
        return count
    print("[ingest] ChromaDB empty — running initial ingest from knowledge/")
    main(reset=False)
    try:
        count = vs._collection.count()
    except Exception:
        count = 0
    print(f"[ingest] ✅ initial ingest complete — {count} chunks available")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="wipe and rebuild")
    args = parser.parse_args()
    main(reset=args.reset)
