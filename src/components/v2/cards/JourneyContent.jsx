import React from 'react';
import Highlightable from '../Highlightable';
import { BodyShell, Section, Label } from './primitives';

const TIMELINE = [
  { id: 'android', year: '2020 – 2022', text: 'Started with Kotlin, Jetpack Compose, and Material 3. Ran 12+ workshops for 50+ developers at GDSC and shipped a campus map app used by 200+ students.' },
  { id: 'ml', year: '2022 – 2024', text: 'Moved into ML and healthcare AI. Published mammography transfer-learning research at 98% accuracy; built a drug-target binding model at Synopsys over 3.2M protein-drug samples.' },
  { id: 'agentic', year: '2024 – now', text: 'At TCS I shipped enterprise systems at real scale. At NYU I now build agentic software where model orchestration, live tools, and UI state all have to feel coherent together — like this site.' },
];

const JourneyContent = ({ mat }) => (
  <BodyShell>
    <Label mat={mat}>Timeline</Label>
    <div style={{ position: 'relative', paddingLeft: 22 }}>
      <div
        style={{
          position: 'absolute',
          left: 5,
          top: 6,
          bottom: 6,
          width: 1,
          background: `${mat.mute}44`,
        }}
      />
      {TIMELINE.map((t) => (
        <Section key={t.id} id={t.id} style={{ marginBottom: 18 }}>
          <div style={{ position: 'relative' }}>
            <div
              style={{
                position: 'absolute',
                left: -21,
                top: 5,
                width: 9,
                height: 9,
                borderRadius: '50%',
                background: mat.accent,
                boxShadow: `0 0 0 3px ${mat.tint}`,
              }}
            />
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                letterSpacing: 1,
                color: mat.mute,
                marginBottom: 4,
              }}
            >
              {t.year}
            </div>
            <div style={{ fontSize: 13, color: mat.ink, opacity: 0.84, lineHeight: 1.55 }}>
              <Highlightable>{t.text}</Highlightable>
            </div>
          </div>
        </Section>
      ))}
    </div>
  </BodyShell>
);

export default JourneyContent;
