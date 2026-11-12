import React from 'react';
import { BodyShell, Section, Row } from './primitives';

const ENTRIES = [
  {
    id: 'gdsc',
    role: 'Android Lead & Google Facilitator',
    org: 'Google Developer Student Club — GGSIPU',
    dates: '2022 – 2023',
    notes: 'Taught 50+ students Jetpack Compose through 12+ workshops, iterating the curriculum week over week based on how people actually learned.',
  },
  {
    id: 'infoxpression',
    role: 'Technical Festival Lead — Infoxpression',
    org: 'GGSIPU',
    dates: '2022 – 2024',
    notes: 'Led 30+ events across a 3-day technical festival, coordinating operations and technical execution for the student body.',
  },
  {
    id: 'hackathons',
    role: 'Hackathon wins',
    org: 'NYC',
    dates: '2025 – 2026',
    notes: '1st Place at Pulse NYC, and Top Technical Build at Google × Columbia for Virtual War Room — the multi-agent voice boardroom.',
  },
];

const LeadershipContent = ({ mat }) => (
  <BodyShell>
    {ENTRIES.map((e) => (
      <Section key={e.id} id={e.id}>
        <Row mat={mat} role={e.role} org={e.org} dates={e.dates} notes={e.notes} />
      </Section>
    ))}
  </BodyShell>
);

export default LeadershipContent;
