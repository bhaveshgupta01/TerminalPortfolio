import React from 'react';
import { Award } from 'lucide-react';
import { Eyebrow, Section, WorkCard, accents } from './primitives';
import Highlightable from '../Highlightable';

const LeadershipCard = () => (
  <div className="space-y-6">
    <Eyebrow icon={Award} color={accents.rose} kicker="Community" title="Leadership &amp; Recognition">
      <Highlightable>
        Where I&apos;ve taught, facilitated, and led — the parts of the journey that weren&apos;t just writing code alone.
      </Highlightable>
    </Eyebrow>

    <div className="space-y-4">
      <Section id="gdsc">
        <WorkCard
          role="Android Lead &amp; Google Facilitator"
          company="Google Developer Student Club — GGSIPU"
          date="2022 – 2023"
          color={accents.emerald}
          desc="Taught 50+ students Jetpack Compose through 12+ workshops, iterating the curriculum week over week based on how people actually learned."
        />
      </Section>
      <Section id="infoxpression">
        <WorkCard
          role="Technical Festival Lead — Infoxpression"
          company="GGSIPU"
          date="2022 – 2024"
          color={accents.amber}
          desc="Led 30+ events across a 3-day technical festival, coordinating both operations and technical execution for the student body."
        />
      </Section>
      <Section id="hackathons">
        <WorkCard
          role="Hackathon Wins"
          company="NYC"
          date="2025 – 2026"
          color={accents.blue}
          desc="1st Place at Pulse NYC, and Top Technical Build at Google × Columbia for Virtual War Room — the multi-agent voice boardroom."
        />
      </Section>
    </div>
  </div>
);

export default LeadershipCard;
