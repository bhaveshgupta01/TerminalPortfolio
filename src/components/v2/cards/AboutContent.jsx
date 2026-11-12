import React from 'react';
import Highlightable from '../Highlightable';
import { BodyShell, Section, Tagline, Para, TagPill } from './primitives';

const AboutContent = ({ mat }) => (
  <BodyShell>
    <Section id="identity">
      <Tagline mat={mat}>I build AI for problems that hold up in the real world.</Tagline>
      <Para mat={mat}>
        I&apos;m a Master&apos;s student at NYU Courant and a Graduate Assistant at NYU Abu Dhabi&apos;s New York Office. My path went Android → ML → domain-specific AI; the through-line is shipping work that survives outside a demo.
      </Para>
      <Para mat={mat}>
        Northern Trust transactions at TCS, drug-target models at Synopsys, Compose workshops for 50+ developers at GDSC, ICAIA-published mammography research, and now agentic interfaces — like the one you&apos;re using right now.
      </Para>
    </Section>

    <Section id="focus">
      <div
        style={{
          display: 'flex',
          gap: 16,
          padding: 14,
          borderRadius: 10,
          background: 'rgba(0,0,0,0.22)',
          border: `1px solid ${mat.mute}33`,
        }}
      >
        <img
          src="/profile_photo.png"
          alt="Bhavesh Gupta"
          style={{
            width: 56,
            height: 56,
            borderRadius: 10,
            objectFit: 'cover',
            border: `1px solid ${mat.mute}55`,
          }}
        />
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9,
              letterSpacing: 1.5,
              color: mat.mute,
              textTransform: 'uppercase',
              marginBottom: 4,
            }}
          >
            Current focus
          </div>
          <div
            style={{
              fontFamily: "'Inter Tight', sans-serif",
              fontSize: 15,
              fontWeight: 600,
              color: mat.ink,
              marginBottom: 4,
            }}
          >
            <Highlightable>Agentic product systems</Highlightable>
          </div>
          <div style={{ fontSize: 12, color: mat.ink, opacity: 0.78, lineHeight: 1.5 }}>
            <Highlightable>
              Voice interfaces, tool-using agents, structured UI responses, and pages that react to what the model is doing.
            </Highlightable>
          </div>
        </div>
      </div>
    </Section>

    <Section id="values">
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 9,
          letterSpacing: 1.5,
          color: mat.mute,
          textTransform: 'uppercase',
          marginBottom: 10,
        }}
      >
        What matters
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {['product sense', 'systems thinking', 'applied AI', 'mobile UX', 'real-world metrics'].map((t) => (
          <TagPill key={t} mat={mat}>{t}</TagPill>
        ))}
      </div>
    </Section>
  </BodyShell>
);

export default AboutContent;
