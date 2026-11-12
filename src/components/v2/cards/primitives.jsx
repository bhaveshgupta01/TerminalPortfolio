import React from 'react';
import Highlightable from '../Highlightable';

/**
 * Body primitives for v2 card detail content. Each takes the parent card's
 * `mat` (material palette) so colours stay coherent within a card and
 * shift between cards. The card.jsx renders these inside a scrollable body
 * area at `padding: 8px 32px 32px` (handled here per primitive instead).
 */

const BODY_PADDING = '8px 28px 28px';

export const BodyShell = ({ children }) => (
  <div style={{ padding: BODY_PADDING }}>{children}</div>
);

export const Section = ({ id, children, style }) => (
  <section
    data-section={id}
    style={{
      padding: 0,
      marginBottom: 14,
      borderRadius: 12,
      transition: 'box-shadow 600ms ease',
      ...style,
    }}
  >
    {children}
  </section>
);

export const Label = ({ mat, children }) => (
  <div
    style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 9,
      letterSpacing: 1.5,
      color: mat.mute,
      textTransform: 'uppercase',
      marginBottom: 8,
    }}
  >
    {children}
  </div>
);

export const Tagline = ({ mat, children }) => (
  <div
    style={{
      fontFamily: "'Inter Tight', sans-serif",
      fontSize: 21,
      fontWeight: 500,
      color: mat.ink,
      lineHeight: 1.3,
      letterSpacing: -0.3,
      marginBottom: 18,
    }}
  >
    <Highlightable>{children}</Highlightable>
  </div>
);

export const Para = ({ mat, children }) => (
  <p
    style={{
      fontSize: 13,
      color: mat.ink,
      opacity: 0.82,
      lineHeight: 1.55,
      marginBottom: 12,
    }}
  >
    <Highlightable>{children}</Highlightable>
  </p>
);

export const Mute = ({ mat, children, style }) => (
  <span style={{ fontSize: 12, color: mat.mute, ...style }}>{children}</span>
);

export const TagPill = ({ mat, children }) => (
  <span
    style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 10,
      letterSpacing: 0.5,
      padding: '5px 10px',
      borderRadius: 999,
      border: `1px solid ${mat.mute}55`,
      color: mat.accent,
    }}
  >
    <Highlightable>{children}</Highlightable>
  </span>
);

export const Row = ({ mat, role, org, dates, notes, accent }) => (
  <div
    style={{
      paddingBottom: 16,
      marginBottom: 16,
      borderBottom: `1px solid ${mat.mute}33`,
    }}
  >
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        marginBottom: 4,
        gap: 12,
      }}
    >
      <div
        style={{
          fontFamily: "'Inter Tight', sans-serif",
          fontSize: 15,
          fontWeight: 600,
          color: mat.ink,
        }}
      >
        <Highlightable>{role}</Highlightable>
      </div>
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 10,
          color: mat.mute,
          whiteSpace: 'nowrap',
        }}
      >
        {dates}
      </div>
    </div>
    <div style={{ color: accent || mat.accent, marginBottom: 6, fontSize: 13 }}>
      <Highlightable>{org}</Highlightable>
    </div>
    <div style={{ fontSize: 13, color: mat.ink, opacity: 0.82, lineHeight: 1.55 }}>
      <Highlightable>{notes}</Highlightable>
    </div>
  </div>
);
