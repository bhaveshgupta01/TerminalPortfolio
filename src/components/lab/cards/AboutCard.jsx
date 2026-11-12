import React from 'react';
import { User } from 'lucide-react';
import LogoIcon from '../../LogoIcon';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, StatTile, accents } from './primitives';

const AboutCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={User} color={accents.rose} kicker="Identity" title="About Me">
      <Highlightable>
        I&apos;m a Master&apos;s student at NYU Courant and a Graduate Assistant at NYU Abu Dhabi&apos;s New York Office, shipping AI systems where the model, the tools, and the interface all have to feel like one product.
      </Highlightable>
    </Eyebrow>

    <Section id="identity" className="space-y-5 rounded-[28px] border-subtle/15 bg-surface-2/60 p-6">
      <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${accents.emerald}`}>
        <span className="h-2 w-2 rounded-full bg-emerald-500" />
        <Highlightable>Open to Summer 2026 roles</Highlightable>
      </div>
      <div>
        <h4 className="text-3xl font-bold tracking-tight text-ink">
          Building AI for
          {' '}
          <span className="bg-gradient-to-r from-rose-400 to-amber-300 bg-clip-text text-transparent">
            problems that matter
          </span>
        </h4>
        <p className="mt-4 text-sm leading-relaxed text-muted">
          <Highlightable>
            My path went Android → ML → domain-specific AI, and the common thread is shipping work that holds up in the real world: Northern Trust transactions at TCS, drug-target models at Synopsys, Compose workshops at GDSC, and now agentic experiences at NYU.
          </Highlightable>
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <StatTile label="Base" value="NYC" sub="MSCS at NYU Courant + NYUAD NY Office" />
        <StatTile label="Core Stack" value="AI + Mobile" sub="React, FastAPI, LangGraph, Gemini, Android" />
        <StatTile label="Mode" value="Impact" sub="Production systems, not portfolio theater" />
      </div>
    </Section>

    <Section id="focus" className="rounded-[28px] border-subtle/15 bg-surface-2/70 p-5">
      <div className="relative flex items-start gap-4">
        <LogoIcon aria-hidden className="absolute -right-2 -top-2 h-24 w-24 text-ink opacity-10" strokeWidth={30} />
        <div className="h-20 w-20 overflow-hidden rounded-2xl bg-surface ring-1 ring-subtle/15">
          <img src="/profile_photo.png" alt="Bhavesh Gupta" className="h-full w-full object-cover" />
        </div>
        <div className="relative z-10">
          <div className="text-xs font-semibold uppercase tracking-[0.22em] text-faint">Current Focus</div>
          <div className="mt-2 text-lg font-bold tracking-tight text-ink">
            <Highlightable>Agentic product systems</Highlightable>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            <Highlightable>
              Voice interfaces, tool-using agents, structured UI responses, and pages that react to what the model is doing — this site is one of them.
            </Highlightable>
          </p>
        </div>
      </div>
    </Section>

    <Section id="values" className="rounded-[28px] border-subtle/15 bg-surface-2/70 p-5">
      <div className="text-xs font-semibold uppercase tracking-[0.22em] text-faint">What matters to me</div>
      <div className="mt-4 flex flex-wrap gap-2">
        {['Product sense', 'Systems thinking', 'Applied AI', 'Mobile UX', 'Real-world metrics'].map((item) => (
          <span key={item} className="rounded-full border border-subtle/15 px-3 py-1.5 text-xs font-medium text-ink">
            <Highlightable>{item}</Highlightable>
          </span>
        ))}
      </div>
    </Section>
  </div>
);

export default AboutCard;
