import React from 'react';
import { Map } from 'lucide-react';
import Highlightable from '../Highlightable';
import { Eyebrow, Section, accents } from './primitives';

const Era = ({ id, era, years, color, text }) => (
  <Section id={id} className="rounded-2xl border-subtle/15 bg-surface-2/65 p-5">
    <div className="flex items-start justify-between gap-3">
      <div className="text-lg font-bold tracking-tight text-ink">
        <Highlightable>{era}</Highlightable>
      </div>
      <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${color}`}>
        {years}
      </span>
    </div>
    <p className="mt-3 text-[13px] leading-relaxed text-muted">
      <Highlightable>{text}</Highlightable>
    </p>
  </Section>
);

const JourneyCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Map} color={accents.amber} kicker="Trajectory" title="My Journey">
      <Highlightable>
        Each era built the constraints for the next one. Android taught me shipping discipline, ML taught me to trust metrics, and agentic systems are letting me combine the two.
      </Highlightable>
    </Eyebrow>

    <div className="space-y-4">
      <Era
        id="android"
        era="Era I — Android"
        years="2020 – 2022"
        color={accents.emerald}
        text="Started with Kotlin, Jetpack Compose, and Material 3. Ran 12+ workshops for 50+ developers at GDSC and shipped a campus map app used by 200+ students."
      />
      <Era
        id="ml"
        era="Era II — ML"
        years="2022 – 2024"
        color={accents.blue}
        text="Moved into machine learning and healthcare AI. Published mammography transfer-learning research (98% accuracy) and built a drug-target binding model at Synopsys over 3.2M protein-drug samples."
      />
      <Era
        id="agentic"
        era="Era III — Agentic AI"
        years="2024 – now"
        color={accents.rose}
        text="At TCS I shipped enterprise systems at real scale. At NYU I now build agentic software where model orchestration, live tools, and UI state all have to feel coherent together — this portfolio being one of those surfaces."
      />
    </div>
  </div>
);

export default JourneyCard;
