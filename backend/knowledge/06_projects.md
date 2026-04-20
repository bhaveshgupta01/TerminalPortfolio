<!-- role: projects -->

# Projects & Achievements

> Scope note: this file is for **personal side projects, hackathon builds, and published research**. It does NOT contain workplace projects from TCS, Synopsys, eSec Forte, GDSC, or NYUAD — those live in their own role-tagged files.

## 1. ICAIA '24 Published Research (Bachelor's thesis, GGSIPU)
**Title:** "Comparative Analysis of Deep Transfer Learning for Mammography Classification"
**Conference:** ICAIA '24 — International Conference on AI in Healthcare
**DOI:** https://doi.org/10.24874/PES.SI.25.03A.007
**Recognition:** Best Presenter Award
Independent academic research from my bachelor's thesis — not affiliated with Synopsys. Demonstrates my depth in applying deep learning to healthcare.

## 2. Virtual War Room (Google x Columbia Hackathon)
**Award:** Top Technical Build across 50+ teams; Google Mentor recognition
**Stack:** Gemini Live API, Google ADK, FastAPI, React, WebSocket
Multi-agent voice boardroom with 4 AI personas. Real-time audio streaming, raise-hand interjection system, document analysis, structured voting, speech synthesis with different AI voices. First-of-its-kind voice-based board meeting simulation.

## 3. Pulse NYC Hackathon — 1st Place (AI Marketing Platform)
**Award:** 1st Place Winner
**Stack:** React, TypeScript, Next.js, AI/ML backend
Event-triggered automated campaign deployment in under 5 seconds. Real-time broadcast event detection, targeted campaign generation, real-time prediction validation.

## 4. VoiceGraph — Voice-First Knowledge Graph (Google x NYU Tandon / Build with AI NYC)
**Co-built with:** Pranjal Mishra, Anzhelika Siui
**Stack:** React 18, TypeScript, Three.js, Python FastAPI, Neo4j AuraDB, Gemini 2.5 Flash, LangChain
**Live:** https://voicegraph-802587268683.us-central1.run.app/

Voice-first interactive knowledge graph. Upload PDFs, YouTube videos, URLs — then talk to the graph via Gemini Live's bidirectional audio while watching AI reason visually across 3D graph topology.

Highlights: 3-phase extraction pipeline (schema-free discovery → OWL ontology → ontology-guided precision extraction with entity resolution); agentic GraphRAG with 8 specialized query tools (semantic search, path finding, community detection, Text2Cypher); real-time Gemini Live with mid-conversation function calling; 3D force-directed graph rendering 500+ nodes in Three.js; WebSocket event protocol multiplexing audio + graph mutations + thinking animations + tool calls; multi-stage Docker build on Cloud Run with Neo4j AuraDB. Live graph: 569 nodes, 882 edges on DC & Energy nexus intelligence.

## 5. GyBuddy — AI Real-Time Fitness Coach (Google AI Hackathon)
**Stack:** React Native, Expo SDK 54, Gemini 2.5 Flash Native Audio Live API, Google Maps APIs, Firebase
Solo project. Mobile fitness app where Gemini acts as your running buddy via real-time voice. Pick a target pace, select a route shape (heart, star, circle), and the AI coaches you through bidirectional audio streaming while GPS tracks you live.

Highlights: WebSocket bidirectional audio (16kHz in → 24kHz out) with 4 voice personalities; 10 custom agent tools (get_current_stats, get_route_info, find_nearby_places, get_weather, generate_route, etc.); smart environment detection (gym proximity via Places API, treadmill auto-detection via pedometer + GPS, weather-aware coaching); parametric equations for heart/star/circle routes; proactive coaching via 15s interval checks; "Ethereal Minimalist" design system with glassmorphism; 65 unit tests across 5 suites with 100% pass rate. ~8,000 lines of TypeScript across 30+ files, 6 Google APIs + 1 weather API, 12 unique prompt combinations.

## 6. SignalFlow — Autonomous AI Crypto Trading Agent
**Stack:** Python, Gemini 2.5 Flash via Vertex AI, Boba Agents MCP (85+ tools), Hyperliquid, Polymarket, Streamlit, Neo4j
Autonomous AI agent trading perpetual futures on Hyperliquid. Monitors prediction markets, whale wallets, and funding rates. Starts with $100 virtual wallet, scans signals, Gemini evaluates, executes with auto stop-loss / take-profit, surfaces on 6-page dashboard.

Pipeline: SCAN (no LLM) → ANALYZE (Gemini + 85 Boba tools) → RISK GATE (hard limits, no LLM override) → EXECUTE (hl_place_order with SL/TP) → MANAGE (auto-close on SL/TP or 4h age) → SNAPSHOT. 6 async triggers at 45s–300s intervals. Event bus via asyncio.Queue. 20+ Boba tools actively used. SQLite persistence with 7 indexed tables. Docker + docker-compose deployment (Railway, GCP). Overnight run: $100 → $105.56 (+5.6%) in 12 hours, 78% win rate (7W/2L), 12 trades across BTC/ETH/SOL.

## 7. MoMA Quest — AI Museum Exploration PWA
**Stack:** React, TypeScript, Next.js, Gemini API, MongoDB/Firestore
PWA that gamifies MoMA exploration. Gemini generates dynamic art-based quests; users complete them by visiting paintings and taking photos. Features: AI-generated runner personas, achievement badges (First Steps, 5K Club, Heart Artist, Early Bird), QR code invitations, weekly leaderboards, pack challenges, 200+ MoMA artworks integrated.

## 8. Agentic RAG Engine — Self-Correcting Retrieval
**Stack:** Python, LangGraph, Mistral 7B (Ollama), ChromaDB, DuckDuckGo, FastAPI
Self-correcting RAG that grades its own retrieval quality. If local documents aren't relevant enough, it falls back to web search. Prevents hallucinations via multi-stage evaluation.

## 9. Healthcare Document RAG
**Stack:** LangGraph, ChromaDB, LLaVA, Mistral 7B
Privacy-preserving medical document Q&A. 6-node agentic retrieval pipeline with 3-tier quality control and LLaVA for document image understanding.

## 10. Super Bowl Ad Viral Post Generator
**Stack:** React, TypeScript, Vite, Gemini API, Next.js
AI marketing tool that detects major events (Super Bowl, Olympics) and generates viral-optimized social posts in real time.

## 11. BunKey — Android Encrypter/Decrypter
**Stack:** Kotlin, Android SDK, Material Design
Custom dual-key encryption algorithm with character-to-integer conversion, segment-based data partitioning, secure key exchange flow, Material Design 3 UI with input validation.

## 12. Other Projects
- **NxSAM_draft** — Dart
- **IPNav** — Kotlin map navigation
- **WallpaperTest** — Android Auto testing
- **Saste Kapade** — Flutter cloth rental app
- **Breast Cancer Detection** — TensorFlow model, 94% accuracy
