import React from 'react';
import { Link } from 'react-router-dom';
import Highlightable from '../Highlightable';
import { designs } from '../../../data/designs';
import { BodyShell, Section, Label } from './primitives';

const FEATURED = designs.slice(0, 6);
const CATEGORY_COUNTS = Object.entries(
  designs.reduce((acc, d) => {
    acc[d.category] = (acc[d.category] || 0) + 1;
    return acc;
  }, {}),
).map(([category, count]) => ({ category, count }));

// Slight stagger angles per thumbnail for the wall effect.
const ROTATIONS = ['-1.6deg', '1deg', '-0.4deg', '1.6deg', '-1.2deg', '0.6deg'];

const DesignContent = ({ mat }) => (
  <BodyShell>
    <Section id="categories">
      <Label mat={mat}>Categories</Label>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4 }}>
        {CATEGORY_COUNTS.map(({ category, count }) => (
          <span
            key={category}
            style={{
              fontSize: 11,
              padding: '4px 10px',
              borderRadius: 999,
              background: 'rgba(0,0,0,0.25)',
              border: `1px solid ${mat.mute}55`,
              color: mat.ink,
            }}
          >
            <Highlightable>{category}</Highlightable>
            <span style={{ marginLeft: 5, color: mat.mute }}>·</span>
            <span style={{ marginLeft: 5, color: mat.accent, fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}>
              {count}
            </span>
          </span>
        ))}
      </div>
    </Section>

    <Section id="featured">
      <Label mat={mat}>Featured</Label>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 10 }}>
        {FEATURED.map((d, i) => (
          <div
            key={d.id}
            style={{
              aspectRatio: '4/5',
              overflow: 'hidden',
              borderRadius: 8,
              border: `1px solid ${mat.mute}55`,
              transform: `rotate(${ROTATIONS[i % ROTATIONS.length]})`,
              transition: 'transform 250ms ease, box-shadow 250ms ease',
              position: 'relative',
              background: 'rgba(0,0,0,0.4)',
              boxShadow: '0 6px 16px -10px rgba(0,0,0,0.55)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'rotate(0deg) scale(1.04)';
              e.currentTarget.style.boxShadow = `0 12px 24px -12px rgba(0,0,0,0.7), 0 0 0 1px ${mat.accent}55`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = `rotate(${ROTATIONS[i % ROTATIONS.length]})`;
              e.currentTarget.style.boxShadow = '0 6px 16px -10px rgba(0,0,0,0.55)';
            }}
          >
            <img
              src={d.image}
              alt={d.title}
              loading="lazy"
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            />
            <div
              style={{
                position: 'absolute',
                inset: 0,
                background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 50%)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-end',
                padding: 8,
              }}
            >
              <div style={{ fontSize: 11, color: mat.ink, fontWeight: 500 }}>
                <Highlightable>{d.title}</Highlightable>
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 8, color: mat.mute, letterSpacing: 0.5 }}>
                {d.category}
              </div>
            </div>
          </div>
        ))}
      </div>
    </Section>

    <div style={{ marginTop: 14 }}>
      <Link
        to="/design"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 14px',
          borderRadius: 999,
          background: mat.accent,
          color: '#1a1410',
          textDecoration: 'none',
          fontFamily: "'Inter Tight', sans-serif",
          fontSize: 12,
          fontWeight: 600,
          boxShadow: `0 1px 0 rgba(255,255,255,0.3) inset, 0 4px 10px ${mat.accent}33`,
        }}
      >
        Open the full gallery →
      </Link>
    </div>
  </BodyShell>
);

export default DesignContent;
