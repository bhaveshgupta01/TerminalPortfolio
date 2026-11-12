import React from 'react';
import { motion } from 'framer-motion';

const SPRING = { type: 'spring', stiffness: 300, damping: 30 };

const WalletCard = ({
  card,
  state,
  isActive,
  isHighlighted,
  isPeekCard,
  hasNextCard,
  onSelect,
  registerScrollRef,
  onCardScroll,
  onCardWheel,
  onTouchStart,
  onTouchEnd,
}) => {
  const Icon = card.icon;
  const Body = card.render;

  return (
    <motion.div
      className={`absolute inset-x-0 top-0 h-full origin-top ${isActive ? 'cursor-default' : 'cursor-pointer'}`}
      animate={{
        y: state.y,
        scale: state.scale,
        opacity: state.opacity,
      }}
      transition={SPRING}
      style={{ zIndex: state.zIndex }}
      onClick={!isActive ? () => onSelect(card.id) : undefined}
    >
      <div
        className={`glass-strong glass-edge relative flex h-full flex-col overflow-hidden rounded-[30px] border transition-colors ${
          isHighlighted ? 'border-accent/40' : 'border-subtle/15'
        }`}
        onClick={isActive ? (e) => e.stopPropagation() : undefined}
      >
        <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${card.gradient}`} />
        <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-white/6 to-transparent dark:from-white/3" />

        <div className="relative flex items-start justify-between gap-4 border-b border-subtle/10 px-5 py-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-subtle/15 bg-surface/60 text-accent">
                <Icon size={18} />
              </div>
              <div className="min-w-0">
                <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-faint">
                  {card.eyebrow}
                </div>
                <h3 className="truncate text-2xl font-bold tracking-tight text-ink">{card.title}</h3>
              </div>
            </div>

            <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted">
              {card.summary}
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              {card.highlights.map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-subtle/15 bg-surface/55 px-3 py-1.5 text-[11px] font-medium text-ink"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>

          <div className="flex shrink-0 flex-col items-end gap-2">
            {isHighlighted && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-full border border-accent/20 bg-accent/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-accent"
              >
                AI focused
              </motion.div>
            )}
            {!isActive && (
              <div className="rounded-full border border-subtle/15 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">
                Tap to open
              </div>
            )}
            {isActive && hasNextCard && (
              <div className="rounded-full border border-subtle/15 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">
                Scroll to move deeper
              </div>
            )}
          </div>
        </div>

        {isActive ? (
          <div
            ref={registerScrollRef(card.id)}
            onScroll={(event) => onCardScroll(card.id, event)}
            onWheel={(event) => onCardWheel(card.id, event)}
            onTouchStart={(event) => onTouchStart(card.id, event)}
            onTouchEnd={(event) => onTouchEnd(card.id, event)}
            className="relative min-h-0 flex-1 overflow-y-auto overscroll-y-contain px-5 pb-24 pt-5 scrollbar-hide"
          >
            <Body />
          </div>
        ) : (
          <div className="relative min-h-0 flex-1 overflow-hidden px-5 pb-6 pt-4">
            <div className="space-y-3 text-sm leading-relaxed text-muted">
              <p>
                {card.summary}
              </p>
              {isPeekCard && (
                <p className="text-faint">
                  Keep scrolling to expand this next card.
                </p>
              )}
            </div>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-surface-2 via-surface-2/90 to-transparent" />
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default WalletCard;
