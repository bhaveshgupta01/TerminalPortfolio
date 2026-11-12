import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ChevronRight, CornerDownLeft } from 'lucide-react';

/**
 * Command palette for jumping to any card, or any sub-section within a card.
 * Entries are pre-built from the LAB_CARDS registry. Keyboard-first:
 *   - ⌘K / Ctrl+K opens
 *   - Esc closes
 *   - ↑ / ↓ move selection
 *   - Enter activates
 */

const buildEntries = (cards) => {
  const list = [];
  cards.forEach((card) => {
    list.push({
      key: `card:${card.id}`,
      cardId: card.id,
      sectionId: null,
      title: card.title,
      subtitle: card.summary,
      breadcrumb: [card.title],
      search: `${card.title} ${card.searchText || ''}`.toLowerCase(),
      Icon: card.icon,
    });
    (card.subsections || []).forEach((sub) => {
      list.push({
        key: `section:${card.id}:${sub.id}`,
        cardId: card.id,
        sectionId: sub.id,
        title: sub.label,
        subtitle: card.title,
        breadcrumb: [card.title, sub.label],
        search: `${card.title} ${sub.label} ${card.searchText || ''}`.toLowerCase(),
        Icon: card.icon,
      });
    });
  });
  return list;
};

const LabCommandPalette = ({ open, onClose, cards, onSelect }) => {
  const entries = useMemo(() => buildEntries(cards), [cards]);
  const [query, setQuery] = useState('');
  const [index, setIndex] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    const tokens = q.split(/\s+/);
    return entries.filter((e) => tokens.every((t) => e.search.includes(t)));
  }, [query, entries]);

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setIndex(0);
    const timer = setTimeout(() => inputRef.current?.focus(), 10);
    return () => clearTimeout(timer);
  }, [open]);

  useEffect(() => {
    if (index >= filtered.length) setIndex(0);
  }, [filtered, index]);

  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${index}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [index]);

  const activate = (entry) => {
    if (!entry) return;
    onSelect(entry);
  };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      activate(filtered[index]);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[90] flex items-start justify-center px-4 pt-24"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onClick={onClose}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -8 }}
            transition={{ duration: 0.16, ease: 'easeOut' }}
            className="glass-strong glass-edge relative w-full max-w-xl overflow-hidden rounded-[28px] border border-subtle/20"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-rose-500/12 via-transparent to-amber-400/8" />
            <div className="relative z-10 flex items-center gap-3 border-b border-subtle/10 px-4 py-3">
              <Search size={16} className="text-muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Jump to a card or section…"
                className="h-10 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-faint"
                aria-label="Search cards and sections"
              />
              <span className="rounded-md border border-subtle/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">
                Esc
              </span>
            </div>

            <div
              ref={listRef}
              className="relative z-10 max-h-[380px] overflow-y-auto pb-2 scrollbar-hide"
              role="listbox"
            >
              {filtered.length === 0 ? (
                <div className="px-4 py-8 text-center text-sm text-muted">
                  No matches.
                </div>
              ) : (
                filtered.map((entry, i) => {
                  const Icon = entry.Icon;
                  const isSel = i === index;
                  return (
                    <button
                      key={entry.key}
                      type="button"
                      data-idx={i}
                      role="option"
                      aria-selected={isSel}
                      onMouseEnter={() => setIndex(i)}
                      onClick={() => activate(entry)}
                      className={`flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                        isSel ? 'bg-accent/10 text-ink' : 'text-muted hover:bg-accent/5'
                      }`}
                    >
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-subtle/15 bg-surface/55 text-accent">
                        <Icon size={15} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.22em] text-faint">
                          {entry.breadcrumb.map((b, k) => (
                            <React.Fragment key={k}>
                              {k > 0 && <ChevronRight size={10} />}
                              <span>{b}</span>
                            </React.Fragment>
                          ))}
                        </div>
                        <div className="truncate text-sm font-semibold text-ink">{entry.title}</div>
                        {entry.subtitle && entry.subtitle !== entry.title && (
                          <div className="truncate text-xs text-faint">{entry.subtitle}</div>
                        )}
                      </div>
                      {isSel && (
                        <CornerDownLeft size={14} className="shrink-0 text-faint" />
                      )}
                    </button>
                  );
                })
              )}
            </div>

            <div className="relative z-10 flex items-center justify-between border-t border-subtle/10 px-4 py-2 text-[10px] uppercase tracking-[0.18em] text-faint">
              <span>Navigate with ↑ ↓</span>
              <span>⏎ to open</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default LabCommandPalette;
