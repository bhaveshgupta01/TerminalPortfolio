"""
Portfolio agent — ReAct style with Gemini function calling.

The model has two tools it can call on its own:
  - search_profile(query):  RAG over Bhavesh's curated corpus (ChromaDB)
  - search_web(query):      DuckDuckGo for live / general info

The model decides when to call each (or both, or neither) based on the
question. Its FINAL message after all tool calls is strict JSON:

  {
    "answer":   "first-person reply",
    "scroll_to": "about|experience|skills|projects|publications|leadership|contact|journey|null",
    "component": null | { "type": "...", "title": "...", "data": {...} },
    "source":   "profile" | "web" | "both"
  }
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "portfolio-profile"

SECTION_IDS = {
    "about", "experience", "skills", "projects",
    "publications", "leadership", "contact", "journey",
}
COMPONENT_TYPES = {
    "stats", "timeline", "cards", "comparison", "list", "code", "quote",
}

# --------------------------------------------------------------------------- #
#                              VECTORSTORE                                    #
# --------------------------------------------------------------------------- #

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

def _vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

# --------------------------------------------------------------------------- #
#                                 TOOLS                                       #
# --------------------------------------------------------------------------- #

KNOWN_ROLES = {
    "personal", "nyuad", "tcs", "synopsys", "esec", "gdsc",
    "skills", "projects", "leadership", "metrics", "publication", "general",
}

# Keywords that force a role filter even if the LLM doesn't pass one.
_ROLE_KEYWORDS: list[tuple[str, str]] = [
    ("tcs", "tcs"), ("tata consultancy", "tcs"), ("northern trust", "tcs"),
    ("custodian bank", "tcs"),
    ("synopsys", "synopsys"), ("drug-target", "synopsys"), ("drug target", "synopsys"),
    ("esec", "esec"), ("e-sec", "esec"), ("esec forte", "esec"),
    ("android auto scanner", "esec"),
    ("gdsc", "gdsc"), ("google developer student", "gdsc"),
    ("compose camp", "gdsc"), ("jetpack compose", "gdsc"),
    ("nyuad", "nyuad"), ("abu dhabi", "nyuad"),
    ("publication", "publication"), ("icaia", "publication"),
    ("mammograph", "publication"),
]

def _autodetect_role(q: str) -> Optional[str]:
    ql = q.lower()
    for needle, role in _ROLE_KEYWORDS:
        if needle in ql:
            return role
    return None

@tool
def search_profile(query: str, role: str = "") -> str:
    """Search Bhavesh's curated profile corpus (resume, projects, experience,
    skills, leadership, publications, metrics, personal philosophy, contact).
    Use this for ANY question about Bhavesh personally.

    Args:
      query: the search query (semantic — use natural language).
      role:  OPTIONAL. Restrict retrieval to ONE role tag. Pass this whenever
             the user's question is scoped to a specific employer, project set,
             or topic. Valid values:
               - "tcs"         (Tata Consultancy Services, Northern Trust client)
               - "synopsys"    (drug-target binding affinity work)
               - "esec"        (eSec Forte Android security internship)
               - "gdsc"        (Google Developer Student Club, Compose Camp)
               - "nyuad"       (NYU Abu Dhabi New York Office, current GA role)
               - "personal"    (about / journey / education / philosophy)
               - "skills"      (technical skills matrix)
               - "projects"    (personal side projects, hackathons, research)
               - "leadership"  (student leadership, mentorship, community)
               - "metrics"     (top-line impact numbers)
               - "publication" (ICAIA '24 mammography paper)
             Leave empty for broad questions that span multiple areas."""
    role = (role or "").strip().lower()
    if role and role not in KNOWN_ROLES:
        role = ""
    if not role:
        detected = _autodetect_role(query)
        if detected:
            role = detected

    try:
        vs = _vectorstore()
        if role:
            docs = vs.similarity_search(query, k=6, filter={"role": role})
            if not docs:  # graceful fallback if nothing matches the filter
                docs = vs.similarity_search(query, k=5)
                role_note = f"(no results for role={role}; showing broad matches)"
            else:
                role_note = f"(filtered to role={role})"
        else:
            docs = vs.similarity_search(query, k=5)
            role_note = "(broad search, no role filter)"
    except Exception as e:
        return f"(profile search failed: {e})"

    if not docs:
        return "(no matching profile content)"

    parts = [role_note]
    for d in docs:
        rtag = d.metadata.get("role", "general")
        parts.append(f"[role={rtag}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

@tool
def search_web(query: str) -> str:
    """Search DuckDuckGo for live / current / general-knowledge info NOT in
    Bhavesh's personal profile. Use for weather, prices, news, definitions of
    concepts, general explanations, or anything time-sensitive. Safe to call
    ALONGSIDE search_profile when a question needs both (e.g. 'what's the
    weather where you live?' → search_profile for location + search_web for
    current weather)."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"(web search failed: {e})"
    if not hits:
        return "(no web results)"
    lines = []
    for h in hits:
        title = h.get("title", "")
        body = h.get("body") or h.get("snippet") or ""
        href = h.get("href") or h.get("url") or ""
        lines.append(f"- {title}\n  {body}\n  {href}")
    return "\n".join(lines)

TOOLS = [search_profile, search_web]

# --------------------------------------------------------------------------- #
#                                  LLM                                        #
# --------------------------------------------------------------------------- #

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
llm_with_tools = llm.bind_tools(TOOLS)

# --------------------------------------------------------------------------- #
#                              SYSTEM PROMPT                                  #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are Bhavesh Gupta's first-person AI twin on his portfolio site.

VOICE
- Always speak in first person as Bhavesh: "I", "my", "me".
- Warm, direct, a little playful. Never corporate.
- Values: selfless, persistent, respectable. Build for real impact, not hype.
- Never invent facts. If unsure, say so plainly.

TOOLS
- search_profile(query): RAG over my own resume/projects/experience/skills/leadership/publications.
- search_web(query):     DuckDuckGo for live / general-knowledge info.

DECISION RULES
- Questions about ME → call search_profile. If the question is scoped to a specific company, project, topic, or publication, PASS the `role` argument to filter retrieval. Examples:
    "what did you do at Synopsys?"    → search_profile(query="work at Synopsys", role="synopsys")
    "tell me about TCS"               → search_profile(query="TCS role and projects", role="tcs")
    "what about eSec Forte?"          → search_profile(query="eSec Forte internship", role="esec")
    "your Compose Camp work?"         → search_profile(query="Compose Camp", role="gdsc")
    "your publication?"               → search_profile(query="ICAIA mammography paper", role="publication")
    "what's your journey?"            → search_profile(query="career journey", role="personal")
    "favorite side projects?"         → search_profile(query="hackathons and side projects", role="projects")
  Omit `role` only for genuinely broad questions (e.g. "who are you?", "what are your strongest skills?").
- Questions needing LIVE info (weather, prices, news, current events, "right now", "today") → call search_web.
- Questions needing BOTH → CALL BOTH TOOLS IN THE SAME TURN (parallel function calls). Examples:
    "what's the weather where you live?"   → search_profile("location of Bhavesh") AND search_web("weather Manhattan New York today")
    "what events are near you this weekend?" → search_profile("Bhavesh location") AND search_web("events New York this weekend")
    "how does my Synopsys stack compare to what's trending now?" → search_profile("Synopsys") AND search_web("trending ML stacks 2026")
- Simple greetings / small talk ("hi", "how are you") → no tools, answer directly.
- Concept explanations with no personal angle ("what is LangGraph?") → search_web.

HARD RULES — DO NOT VIOLATE
- NEVER say "let me check", "I'll look that up", "one moment", or any promise to fetch data UNLESS you are actually emitting a tool call in the same message. If you need data, CALL THE TOOL. Don't announce — act.
- NEVER return a JSON "answer" that makes a promise to search later. The JSON is the FINAL answer. If live data is needed, you must have already fetched it via a tool call earlier in this same request.
- If a tool returned "(no results)" or a failure, acknowledge that honestly in the answer (e.g. "I couldn't pull live weather just now, but I'm in Manhattan").

NO CROSS-ROLE ATTRIBUTION — CRITICAL
Every profile chunk returned by search_profile is tagged with "[role=X]". When you narrate facts:
- ONLY use facts from chunks whose role matches the user's question scope.
- If a user asked about Synopsys, do NOT pull facts from a [role=tcs] chunk into the Synopsys answer.
- If retrieved context spans multiple roles and the user asked about one, ignore the others.
- When in doubt about attribution, say less rather than more. Factual precision beats completeness.
- The "GenAI data validation pipeline" belongs to TCS ONLY. The "drug-target binding model" belongs to Synopsys ONLY. The "98% mammography" figure belongs to the ICAIA publication ONLY — never to Synopsys. The "Android Auto security scanner + Flutter POC" belongs to eSec Forte ONLY — "BunKey" is a separate personal project.

You may call tools multiple times if the first result is insufficient. Prefer FEWER, more specific queries over many vague ones.

FINAL RESPONSE FORMAT
Your LAST message (after all tool calls resolve) MUST be STRICT JSON, no markdown fences, no prose before/after:

{
  "answer": "first-person conversational reply — 2-5 sentences typically, longer only if the question demands depth",
  "scroll_to": "about" | "experience" | "skills" | "projects" | "publications" | "leadership" | "contact" | "journey" | null,
  "component": null | {
      "type": "stats" | "timeline" | "cards" | "comparison" | "list" | "code" | "quote",
      "title": "string",
      "data": object
  },
  "source": "profile" | "web" | "both"
}

FIELD GUIDANCE
- scroll_to: section on the portfolio most relevant to your answer. Use null if nothing fits.
- component: include ONLY when a visual genuinely aids understanding:
    stats      → 3+ quantitative datapoints  (items: [{label,value,sub}])
    timeline   → chronology                   (items: [{date,title,desc}])
    cards      → 2-4 parallel things          (items: [{title,desc,tags}])
    comparison → 2 things side by side        (left:{title,points[]}, right:{title,points[]})
    list       → enumeration                  (items: [str])
    code       → code sample                  (language, code)
    quote      → pull quote                   (text, attribution)
  Otherwise set component to null. Don't force a component.
- source: "profile" if only search_profile was used, "web" if only search_web, "both" if both, else "profile"."""

# --------------------------------------------------------------------------- #
#                                 GRAPH                                       #
# --------------------------------------------------------------------------- #

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState) -> dict:
    """LLM turn. Prepends the system prompt fresh each call (not stored in state)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"

tool_node = ToolNode(TOOLS)

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --------------------------------------------------------------------------- #
#                            OUTPUT PARSING                                    #
# --------------------------------------------------------------------------- #

def parse_output(raw: str | list | None) -> dict:
    """Parse the agent's final JSON message into a validated response dict."""
    if raw is None:
        text = ""
    elif isinstance(raw, list):
        # langchain-google-genai sometimes returns content as a list of parts
        text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        ).strip()
    else:
        text = str(raw).strip()

    # strip markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except Exception:
        return {
            "answer": text or "I'm not sure how to answer that.",
            "scroll_to": None,
            "component": None,
            "source": "profile",
        }

    scroll_to = data.get("scroll_to")
    data["scroll_to"] = scroll_to if scroll_to in SECTION_IDS else None

    comp = data.get("component")
    if isinstance(comp, dict) and comp.get("type") in COMPONENT_TYPES:
        data["component"] = comp
    else:
        data["component"] = None

    data.setdefault("answer", "")
    data.setdefault("source", "profile")
    return data

def run(question: str, thread_id: str) -> dict:
    """Public helper — invoke the agent for one user turn."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    last = result["messages"][-1]
    return parse_output(getattr(last, "content", ""))

# --------------------------------------------------------------------------- #
#                                  CLI                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Portfolio agent (ReAct, Gemini tool-calling). Ctrl-C to quit.\n")
    tid = "cli"
    while True:
        try:
            q = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        out = run(q, tid)
        print("\nbhavesh:", out.get("answer", ""))
        if out.get("scroll_to"):
            print(f"  ↳ scroll: {out['scroll_to']}")
        if out.get("component"):
            print(f"  ↳ component: {out['component'].get('type')} — {out['component'].get('title')}")
        print(f"  ↳ source: {out.get('source')}\n")
