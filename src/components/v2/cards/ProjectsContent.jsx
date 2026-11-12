import React from 'react';
import Highlightable from '../Highlightable';
import { BodyShell, Section } from './primitives';

const PROJECTS = [
  {
    id: 'voicegraph',
    name: 'VoiceGraph',
    year: '2025',
    stack: 'Gemini Live · Neo4j · FastAPI',
    role: 'Solo',
    desc: 'Voice-first interactive knowledge graph. Upload PDFs/URLs, then talk to a 3D graph of nodes via Gemini Live audio with mid-conversation function calling.',
    outcome: '569 nodes · 882 edges · live on Cloud Run',
    link: 'https://voicegraph-802587268683.us-central1.run.app/',
  },
  {
    id: 'gybuddy',
    name: 'GyBuddy',
    year: '2025',
    stack: 'React Native · Gemini Live · Expo',
    role: 'Solo',
    desc: 'AI running companion using Gemini 2.5 Flash Native Audio for real-time coaching, route awareness, and treadmill detection.',
    outcome: 'Production iOS / Android build',
    link: 'https://github.com/bhaveshgupta01',
  },
  {
    id: 'signalflow',
    name: 'SignalFlow',
    year: '2025',
    stack: 'Agentic AI · MCP · Hyperliquid',
    role: 'Solo',
    desc: 'Autonomous crypto trading agent coordinating async market signals + wallet activity over a large Gemini-evaluated tool surface.',
    outcome: 'Live signal pipeline',
    link: 'https://github.com/bhaveshgupta01',
  },
  {
    id: 'virtual-war-room',
    name: 'Virtual War Room',
    year: '2025',
    stack: 'Google ADK · Gemini Live · WebSocket',
    role: 'Hackathon team',
    desc: 'Multi-agent voice boardroom with 4 AI personas, raise-hand interjections, structured voting.',
    outcome: 'Top Technical Build · Google × Columbia',
    link: 'https://github.com/bhaveshgupta01',
  },
  {
    id: 'pulse-nyc',
    name: 'Pulse NYC',
    year: '2025',
    stack: 'Next.js · TypeScript · Realtime',
    role: 'Hackathon team',
    desc: 'AI marketing prototype with event-triggered campaign deployment and real-time audience targeting.',
    outcome: '1st place · Pulse NYC',
    link: 'https://github.com/bhaveshgupta01',
  },
  {
    id: 'agentic-rag',
    name: 'Agentic RAG Engine',
    year: '2026',
    stack: 'LangGraph · ChromaDB · Gemini',
    role: 'Solo',
    desc: 'Self-correcting RAG engine that grades retrieval quality and falls back to web search. Powers this portfolio agent.',
    outcome: 'You are using it right now.',
    link: 'https://github.com/bhaveshgupta01',
  },
];

const ProjectTile = ({ p, mat }) => (
  <a
    href={p.link}
    target="_blank"
    rel="noopener noreferrer"
    style={{
      display: 'block',
      padding: 14,
      borderRadius: 10,
      background: 'rgba(0,0,0,0.28)',
      border: `1px solid ${mat.mute}33`,
      color: 'inherit',
      textDecoration: 'none',
      transition: 'border-color 200ms ease, background 200ms ease',
    }}
    onMouseEnter={(e) => { e.currentTarget.style.borderColor = `${mat.accent}66`; }}
    onMouseLeave={(e) => { e.currentTarget.style.borderColor = `${mat.mute}33`; }}
  >
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
      <div style={{ fontFamily: "'Inter Tight', sans-serif", fontSize: 14, fontWeight: 600, color: mat.ink }}>
        <Highlightable>{p.name}</Highlightable>
      </div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: mat.mute }}>{p.year}</div>
    </div>
    <div style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 9,
      letterSpacing: 0.5,
      color: mat.accent,
      marginBottom: 8,
    }}>
      <Highlightable>{p.stack}</Highlightable> · {p.role}
    </div>
    <div style={{ fontSize: 12, color: mat.ink, opacity: 0.82, lineHeight: 1.5, marginBottom: 10 }}>
      <Highlightable>{p.desc}</Highlightable>
    </div>
    <div style={{
      fontSize: 11,
      color: mat.accent,
      padding: '6px 10px',
      borderRadius: 6,
      background: `${mat.accent}11`,
      border: `1px solid ${mat.accent}22`,
      fontFamily: "'JetBrains Mono', monospace",
      letterSpacing: 0.3,
    }}>
      → <Highlightable>{p.outcome}</Highlightable>
    </div>
  </a>
);

const ProjectsContent = ({ mat }) => (
  <BodyShell>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
      {PROJECTS.map((p) => (
        <Section key={p.id} id={p.id} style={{ marginBottom: 0 }}>
          <ProjectTile p={p} mat={mat} />
        </Section>
      ))}
    </div>
  </BodyShell>
);

export default ProjectsContent;
