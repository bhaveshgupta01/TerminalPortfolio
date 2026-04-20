import React from 'react';

/**
 * Renders a structured component returned by the agent.
 *
 * Shape:
 *   { type: "stats"|"timeline"|"cards"|"comparison"|"list"|"code"|"quote",
 *     title: string,
 *     data:  {...} }
 */
// Gemini sometimes emits `data: [...]` directly instead of `data: { items: [...] }`.
// Accept either shape.
const asItems = (data) =>
  Array.isArray(data) ? data : Array.isArray(data?.items) ? data.items : [];

const DynamicComponent = ({ component }) => {
  if (!component || !component.type) return null;
  const { type, title, data = {} } = component;

  const Shell = ({ children }) => (
    <div className="bg-surface-2 border border-subtle/15 rounded-xl p-4 mt-1">
      {title && (
        <div className="text-xs uppercase tracking-wider text-faint font-bold mb-3">
          {title}
        </div>
      )}
      {children}
    </div>
  );

  switch (type) {
    case 'stats':
      return (
        <Shell>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {asItems(data).map((it, i) => (
              <div key={i} className="bg-surface rounded-lg p-3 border border-subtle/15">
                <div className="text-lg md:text-xl font-bold text-accent">{it.value}</div>
                <div className="text-xs text-muted font-medium mt-0.5">{it.label}</div>
                {it.sub && <div className="text-[10px] text-faint mt-1">{it.sub}</div>}
              </div>
            ))}
          </div>
        </Shell>
      );

    case 'timeline':
      return (
        <Shell>
          <ol className="relative border-l border-subtle/15 ml-2 space-y-4">
            {asItems(data).map((it, i) => (
              <li key={i} className="pl-4">
                <div className="absolute -left-1.5 w-3 h-3 bg-accent rounded-full border-2 border-surface-2"></div>
                <div className="text-[10px] uppercase tracking-wider text-accent-2 font-bold">{it.date}</div>
                <div className="text-sm font-bold text-ink mt-0.5">{it.title}</div>
                <div className="text-xs text-muted mt-1 leading-relaxed">{it.desc}</div>
              </li>
            ))}
          </ol>
        </Shell>
      );

    case 'cards':
      return (
        <Shell>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {asItems(data).map((it, i) => (
              <div key={i} className="bg-surface rounded-lg p-3 border border-subtle/15">
                <div className="text-sm font-bold text-ink mb-1">{it.title}</div>
                <div className="text-xs text-muted leading-relaxed">{it.desc}</div>
                {it.tags?.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {it.tags.map((t, j) => (
                      <span key={j} className="text-[10px] px-2 py-0.5 rounded border border-subtle/15 text-muted">{t}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Shell>
      );

    case 'comparison': {
      // Agent sometimes emits { data: { left, right } }, sometimes { left, right }
      // at the component top level. Accept either.
      const left  = data?.left  ?? component.left  ?? {};
      const right = data?.right ?? component.right ?? {};
      return (
        <Shell>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[['left', left], ['right', right]].map(([side, block]) => (
              <div key={side} className="bg-surface rounded-lg p-3 border border-subtle/15">
                <div className="text-sm font-bold text-ink mb-2">{block.title}</div>
                <ul className="space-y-1">
                  {(block.points || []).map((p, i) => (
                    <li key={i} className="text-xs text-muted flex gap-2">
                      <span className="text-accent shrink-0">•</span>
                      <span className="leading-relaxed">{p}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Shell>
      );
    }

    case 'list':
      return (
        <Shell>
          <ul className="space-y-1.5">
            {asItems(data).map((it, i) => (
              <li key={i} className="text-xs text-ink flex gap-2">
                <span className="text-accent shrink-0">▸</span>
                <span className="leading-relaxed">{it}</span>
              </li>
            ))}
          </ul>
        </Shell>
      );

    case 'code':
      return (
        <Shell>
          <pre className="bg-surface border border-subtle/15 rounded-lg p-3 overflow-x-auto">
            <code className="text-xs text-emerald-600 dark:text-emerald-300 font-mono whitespace-pre">{data.code}</code>
          </pre>
          {data.language && (
            <div className="text-[10px] text-faint mt-1.5 font-mono">{data.language}</div>
          )}
        </Shell>
      );

    case 'quote':
      return (
        <Shell>
          <blockquote className="border-l-2 border-accent pl-3 italic text-muted text-sm leading-relaxed">
            "{data.text}"
            {data.attribution && (
              <div className="mt-2 text-xs text-faint not-italic">— {data.attribution}</div>
            )}
          </blockquote>
        </Shell>
      );

    default:
      return null;
  }
};

export default DynamicComponent;
