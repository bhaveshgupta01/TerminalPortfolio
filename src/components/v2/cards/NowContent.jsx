import React, { useEffect, useState } from 'react';
import Highlightable from '../Highlightable';
import { BodyShell, Section } from './primitives';

/**
 * "What I'm building this week" — the only card that pulls live content
 * from the backend (`GET /now`). The backend reads backend/now.json so I
 * can update without redeploying. Falls back to a sensible static line
 * list if the request fails (e.g. offline backend).
 */

const API_URL = import.meta.env.VITE_AGENT_API || 'http://localhost:8000';

const FALLBACK = {
  updated: 'Right now',
  lines: [
    { icon: '▲', text: "Shipping the v2 wallet — you're looking at it." },
    { icon: '◆', text: 'MSCS coursework at NYU Courant.' },
    { icon: '✦', text: 'Open to Summer 2026 internships.' },
  ],
};

const NowContent = ({ mat }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    fetch(`${API_URL}/now`, { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((j) => setData(j))
      .catch(() => { setError(true); setData(FALLBACK); });
    return () => ctrl.abort();
  }, []);

  const view = data || FALLBACK;

  return (
    <BodyShell>
      <Section id="updates">
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            letterSpacing: 1.5,
            color: mat.mute,
            textTransform: 'uppercase',
            marginBottom: 14,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: mat.accent,
              boxShadow: `0 0 8px ${mat.accent}aa`,
            }}
          />
          {view.updated}
          {error && <span style={{ marginLeft: 8, opacity: 0.6 }}>· offline · showing fallback</span>}
        </div>

        {view.lines.map((l, i) => (
          <div
            key={i}
            style={{
              display: 'flex',
              gap: 14,
              alignItems: 'flex-start',
              padding: '12px 0',
              borderBottom: i < view.lines.length - 1 ? `1px solid ${mat.mute}22` : 'none',
            }}
          >
            <span
              style={{
                color: mat.accent,
                fontSize: 12,
                width: 14,
                flexShrink: 0,
                textAlign: 'center',
                marginTop: 2,
              }}
            >
              {l.icon}
            </span>
            <span style={{
              flex: 1,
              fontSize: 13,
              color: mat.ink,
              opacity: 0.85,
              lineHeight: 1.55,
            }}>
              <Highlightable>{l.text}</Highlightable>
            </span>
          </div>
        ))}
      </Section>
    </BodyShell>
  );
};

export default NowContent;
