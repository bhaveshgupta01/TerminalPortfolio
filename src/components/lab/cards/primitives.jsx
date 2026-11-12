import React from 'react';
import Highlightable from '../Highlightable';

export const accents = {
  rose: 'text-rose-500 bg-rose-500/10 border-rose-500/20',
  amber: 'text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20',
  emerald: 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  blue: 'text-blue-600 dark:text-blue-400 bg-blue-500/10 border-blue-500/20',
  purple: 'text-violet-600 dark:text-violet-400 bg-violet-500/10 border-violet-500/20',
};

/**
 * Wraps a sub-section of card content so the lab can scroll to it and pulse
 * a highlight ring around it when the AI (or the command palette) targets it.
 */
export const Section = ({ id, children, className = '' }) => (
  <section
    data-section={id}
    id={`lab-section-${id}`}
    className={`scroll-mt-4 rounded-[28px] border border-subtle/0 transition-[border-color,box-shadow] duration-500 ${className}`}
  >
    {children}
  </section>
);

export const Eyebrow = ({ icon: Icon, color, kicker, title, children }) => (
  <div className="space-y-3">
    <div className="flex flex-wrap items-center gap-3">
      {Icon && (
        <div className={`flex h-11 w-11 items-center justify-center rounded-2xl border ${color}`}>
          <Icon size={18} />
        </div>
      )}
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-faint">
          {kicker}
        </div>
        <h3 className="text-2xl font-bold tracking-tight text-ink">{title}</h3>
      </div>
    </div>
    {children && <p className="max-w-3xl text-sm leading-relaxed text-muted">{children}</p>}
  </div>
);

export const StatTile = ({ label, value, sub }) => (
  <div className="rounded-2xl border border-subtle/15 bg-surface-2/70 p-4">
    <div className="text-lg font-bold text-ink">
      <Highlightable>{value}</Highlightable>
    </div>
    <div className="mt-1 text-xs font-medium uppercase tracking-[0.22em] text-faint">{label}</div>
    {sub && (
      <div className="mt-2 text-xs leading-relaxed text-muted">
        <Highlightable>{sub}</Highlightable>
      </div>
    )}
  </div>
);

export const WorkCard = ({ role, company, date, desc, color }) => (
  <div className="rounded-2xl border border-subtle/15 bg-surface-2/65 p-5">
    <div className="flex items-start justify-between gap-3">
      <div>
        <h4 className="text-base font-bold tracking-tight text-ink">
          <Highlightable>{role}</Highlightable>
        </h4>
        <p className="mt-1 text-sm font-medium text-muted">
          <Highlightable>{company}</Highlightable>
        </p>
      </div>
      <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${color}`}>
        {date}
      </span>
    </div>
    <p className="mt-4 text-[13px] leading-relaxed text-faint">
      <Highlightable>{desc}</Highlightable>
    </p>
  </div>
);

export const Chip = ({ children, color = '' }) => (
  <span className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${color || 'border-subtle/15 text-ink'}`}>
    <Highlightable>{children}</Highlightable>
  </span>
);
