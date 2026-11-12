# Changelog

All notable stable releases for this project are tracked here.

## [0.2.0] - 2026-11-12

New home page built around three scroll-driven lines (IPU, TCS, NYU).

- `/` is the new page; the original site moved to `/v1`; `/v2`, `/wallet`, `/lab` and `/design` are unchanged.
- Line engine: poses measured from the page, spring physics, cursor interaction, opening draw-in.
- Projects index with filters and detail cards, collage gallery with lightbox, new contact screen.
- Chat moved to a single Vercel function (`api/chat.js`) on the Claude API; the Railway backend is retired but kept in `backend/`.
- Visitor questions are recorded; optional Redis database for a durable log and shared rate limit.
- Old versions are lazy-loaded; share image, favicon and theme-before-paint added.

## [0.1.0] - 2026-04-20

Initial stable baseline for the current AI-first portfolio.

- React 19 + Vite frontend with portfolio pages, theme system, ambient motion, and Dynamic Island chat UI.
- FastAPI + LangGraph backend with profile RAG, web search, structured components, and knowledge ingest flow.
- Local setup and deployment docs for the frontend and backend.
- Stable rollback point to be tagged in git as `v0.1.0`.
