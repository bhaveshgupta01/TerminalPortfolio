import React from 'react';
import { Folder, ExternalLink } from 'lucide-react';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, accents } from './primitives';

const Project = ({ id, title, desc, tags, color, link }) => (
  <Section id={id}>
    <a
      href={link}
      target="_blank"
      rel="noopener noreferrer"
      className="group block rounded-2xl border border-subtle/15 bg-surface-2/65 p-5 transition-colors hover:border-accent/25"
    >
      <div className="flex items-start justify-between gap-4">
        <h4 className="text-base font-bold tracking-tight text-ink group-hover:text-accent">
          <Highlightable>{title}</Highlightable>
        </h4>
        <ExternalLink size={16} className="shrink-0 text-faint transition-colors group-hover:text-accent" />
      </div>
      <p className="mt-3 text-[13px] leading-relaxed text-muted">
        <Highlightable>{desc}</Highlightable>
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span key={tag} className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${color}`}>
            <Highlightable>{tag}</Highlightable>
          </span>
        ))}
      </div>
    </a>
  </Section>
);

const ProjectsCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Folder} color={accents.blue} kicker="Builds" title="Featured Projects">
      <Highlightable>
        The highest-signal work I&apos;ve shipped — from a voice-first knowledge graph to an AI running companion to multi-agent voice systems.
      </Highlightable>
    </Eyebrow>

    <div className="grid gap-4 lg:grid-cols-2">
      <Project
        id="voicegraph"
        title="VoiceGraph"
        desc="Voice-first interactive knowledge graph with Gemini Live audio, Three.js graph rendering, and GraphRAG-style navigation over 569 nodes and 882 edges."
        tags={['Gemini Live', 'LangChain', 'Neo4j', 'Three.js', 'FastAPI']}
        color={accents.purple}
        link="https://voicegraph-802587268683.us-central1.run.app/"
      />
      <Project
        id="gybuddy"
        title="GyBuddy"
        desc="AI running companion on React Native using Gemini 2.5 Flash Native Audio for real-time coaching, route awareness, and treadmill detection."
        tags={['React Native', 'Expo', 'Gemini Live', 'Maps API', 'Firebase']}
        color={accents.amber}
        link="https://github.com/bhaveshgupta01"
      />
      <Project
        id="signalflow"
        title="SignalFlow"
        desc="Autonomous crypto trading agent coordinating asynchronous market signals, wallet activity, and Gemini-based evaluation over a large tool surface."
        tags={['Agentic AI', 'Gemini Vertex', 'MCP', 'Hyperliquid', 'Streamlit']}
        color={accents.rose}
        link="https://github.com/bhaveshgupta01"
      />
      <Project
        id="virtual-war-room"
        title="Virtual War Room"
        desc="Top Technical Build at Google × Columbia: a multi-agent voice boardroom with four AI personas, raise-hand interjections, and structured voting."
        tags={['Google ADK', 'Gemini Live', 'FastAPI', 'React', 'WebSocket']}
        color={accents.blue}
        link="https://github.com/bhaveshgupta01"
      />
      <Project
        id="pulse-nyc"
        title="Pulse NYC"
        desc="1st place AI marketing prototype with event-triggered campaign deployment, real-time audience targeting, and rapid validation loops."
        tags={['Next.js', 'TypeScript', 'AI', 'Real-time']}
        color={accents.emerald}
        link="https://github.com/bhaveshgupta01"
      />
      <Project
        id="agentic-rag"
        title="Agentic RAG Engine"
        desc="Self-correcting RAG engine that grades retrieval quality and falls back to web search when local knowledge is weak. This powers the current portfolio agent."
        tags={['LangGraph', 'ChromaDB', 'Gemini', 'RAG']}
        color={accents.purple}
        link="https://github.com/bhaveshgupta01"
      />
    </div>
  </div>
);

export default ProjectsCard;
