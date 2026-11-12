import { useEffect, useRef, useState } from 'react';

const STARTERS = [
  'What did you build at TCS?',
  'Which hackathons have you won?',
  'Show me VoiceGraph',
];

const OFFLINE = "I can't reach my assistant right now. Everything is on the page, or email me at bg2896@nyu.edu.";

/**
 * The chat that stays with you down the page. Answers can move the page:
 * `onNavigate(sectionId, projectId)` scrolls to a section and opens a project.
 */
export default function ChatDock({ onNavigate }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [focused, setFocused] = useState(false);
  const logRef = useRef(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy, open]);

  const send = async (text) => {
    const question = (text ?? input).trim();
    if (!question || busy) return;
    const history = messages.filter((m) => !m.error).map((m) => ({ role: m.role, text: m.text }));
    setMessages((prev) => [...prev, { role: 'user', text: question }]);
    setInput('');
    setOpen(true);
    setBusy(true);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, history }),
      });
      const data = await res.json().catch(() => ({}));
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer || OFFLINE, component: data.component || null, error: !res.ok || !!data.error },
      ]);
      if (res.ok && (data.section || data.project)) onNavigate?.(data.section, data.project);
    } catch {
      setMessages((prev) => [...prev, { role: 'assistant', text: OFFLINE, error: true }]);
    } finally {
      setBusy(false);
    }
  };

  const hasLog = messages.length > 0;

  return (
    <div className="st-dock">
      {hasLog && open && (
        <div className="st-dock-log" ref={logRef} aria-live="polite">
          {messages.map((m, i) => (
            <div key={i} className={`st-msg st-msg-${m.role}`}>
              <p>{m.text}</p>
              {m.component && <Component c={m.component} />}
            </div>
          ))}
          {busy && <div className="st-msg st-msg-assistant st-msg-wait"><p>…</p></div>}
        </div>
      )}

      {!hasLog && focused && (
        <div className="st-dock-starters">
          {STARTERS.map((s) => (
            <button key={s} type="button" onMouseDown={(e) => { e.preventDefault(); send(s); }}>{s}</button>
          ))}
          <p>An AI answers as me, from my profile. Questions are saved so I can see what people ask.</p>
        </div>
      )}

      <form
        className="st-dock-bar"
        onSubmit={(e) => { e.preventDefault(); send(); }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask me anything about my work"
          aria-label="Ask a question"
          maxLength={600}
        />
        {hasLog && (
          <button type="button" className="st-dock-toggle" onClick={() => setOpen((v) => !v)}>
            {open ? 'Hide' : 'Show'}
          </button>
        )}
        <button type="submit" className="st-dock-send" disabled={busy || !input.trim()}>Ask</button>
      </form>
    </div>
  );
}

function Component({ c }) {
  const items = c.items || [];
  return (
    <div className={`st-comp st-comp-${c.type}`}>
      {c.title && <div className="st-comp-title">{c.title}</div>}
      {c.type === 'stats' && (
        <div className="st-comp-stats">
          {items.map((it, i) => (
            <div key={i}><strong>{it.title}</strong><span>{it.detail}</span></div>
          ))}
        </div>
      )}
      {c.type === 'timeline' && (
        <ol>
          {items.map((it, i) => (
            <li key={i}><em>{it.meta}</em><strong>{it.title}</strong>{it.detail && <span>{it.detail}</span>}</li>
          ))}
        </ol>
      )}
      {c.type === 'cards' && (
        <div className="st-comp-cards">
          {items.map((it, i) => (
            <div key={i}><strong>{it.title}</strong><span>{it.detail}</span>{it.meta && <em>{it.meta}</em>}</div>
          ))}
        </div>
      )}
      {c.type === 'list' && (
        <ul>{items.map((it, i) => <li key={i}>{it.title}{it.detail ? `: ${it.detail}` : ''}</li>)}</ul>
      )}
    </div>
  );
}
