import React, { useState } from 'react';
import Highlightable from '../Highlightable';
import { downloadVCard } from '../../../lib/vcard';
import { BodyShell, Section, Label } from './primitives';

const HANDLES = [
  { label: 'GitHub', value: '@bhaveshgupta01', url: 'https://github.com/bhaveshgupta01' },
  { label: 'LinkedIn', value: '/in/bhaveshgupta01', url: 'https://linkedin.com/in/bhaveshgupta01' },
  { label: 'Web', value: 'libralpanda.vercel.app', url: 'https://libralpanda.vercel.app' },
];

const QRBlock = ({ mat }) => (
  <div
    style={{
      width: 130,
      height: 130,
      flexShrink: 0,
      borderRadius: 10,
      overflow: 'hidden',
      background: '#fff',
      padding: 8,
    }}
  >
    <img
      src="/qrcode1.jpeg"
      alt="QR for vCard"
      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
    />
  </div>
);

const ContactContent = ({ mat }) => {
  const [copied, setCopied] = useState(false);

  const copyEmail = (e) => {
    e.stopPropagation();
    navigator.clipboard?.writeText('bg2896@nyu.edu');
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <BodyShell>
      <Section id="vcard">
        <div style={{ display: 'flex', gap: 22, alignItems: 'flex-start', marginBottom: 18 }}>
          <QRBlock mat={mat} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <Label mat={mat}>Email</Label>
            <button
              type="button"
              onClick={copyEmail}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
                padding: '10px 12px',
                marginBottom: 10,
                background: `${mat.accent}15`,
                border: `1px solid ${mat.accent}44`,
                borderRadius: 8,
                color: mat.ink,
                fontSize: 13,
                cursor: 'pointer',
                fontFamily: "'Inter Tight', sans-serif",
              }}
            >
              <span style={{ wordBreak: 'break-all' }}>
                <Highlightable>bg2896@nyu.edu</Highlightable>
              </span>
              <span
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 10,
                  letterSpacing: 1,
                  color: mat.accent,
                  textTransform: 'uppercase',
                  whiteSpace: 'nowrap',
                  marginLeft: 8,
                }}
              >
                {copied ? 'copied ✓' : 'copy'}
              </span>
            </button>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); downloadVCard(); }}
              style={{
                display: 'block',
                width: '100%',
                padding: '10px 12px',
                marginBottom: 4,
                background: mat.accent,
                color: '#1a1410',
                borderRadius: 8,
                border: 'none',
                fontFamily: "'Inter Tight', sans-serif",
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
                boxShadow: `0 1px 0 rgba(255,255,255,0.3) inset, 0 2px 8px ${mat.accent}55`,
              }}
            >
              Download vCard ↓
            </button>
            <a
              href="mailto:bg2896@nyu.edu"
              style={{
                display: 'block',
                textAlign: 'center',
                padding: '10px 12px',
                marginTop: 8,
                background: 'rgba(0,0,0,0.25)',
                color: mat.ink,
                borderRadius: 8,
                border: `1px solid ${mat.mute}55`,
                fontFamily: "'Inter Tight', sans-serif",
                fontSize: 13,
                textDecoration: 'none',
              }}
            >
              Write an email →
            </a>
          </div>
        </div>
      </Section>

      <Section id="channels">
        <Label mat={mat}>Elsewhere</Label>
        {HANDLES.map((h) => (
          <a
            key={h.label}
            href={h.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 0',
              borderBottom: `1px solid ${mat.mute}22`,
              gap: 10,
              textDecoration: 'none',
              color: 'inherit',
            }}
          >
            <span style={{ fontSize: 12, color: mat.mute }}>{h.label}</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                fontSize: 12,
                color: mat.ink,
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                {h.value}
              </span>
              <span style={{ color: mat.accent, fontSize: 11 }}>↗</span>
            </span>
          </a>
        ))}
      </Section>
    </BodyShell>
  );
};

export default ContactContent;
