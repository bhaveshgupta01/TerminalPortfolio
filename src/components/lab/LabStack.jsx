import React, { useCallback, useEffect, useRef, useState } from 'react';
import LabCard from './LabCard';

const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

const getMetrics = (node) => {
  const max = Math.max(node.scrollHeight - node.clientHeight, 0);
  const progress = max > 0 ? node.scrollTop / max : 0;
  return {
    atTop: node.scrollTop <= 4,
    atBottom: node.scrollTop >= max - 4,
    progress,
  };
};

const LabStack = ({
  cards,
  activeCard,
  highlightedCard,
  focusedSection,
  highlightPulseKey,
  onSelectCard,
  onClearActiveCard,
}) => {
  const containerRef = useRef(null);
  const contentRefs = useRef({});
  const touchStartYRef = useRef(null);
  const [containerHeight, setContainerHeight] = useState(480);
  const [scrollState, setScrollState] = useState({ progress: 0, atTop: true, atBottom: false });

  const activeIndex = cards.findIndex((c) => c.id === activeCard);
  const nextCard = activeIndex >= 0 ? cards[activeIndex + 1] : null;
  const prevCard = activeIndex > 0 ? cards[activeIndex - 1] : null;

  const peekProgress = activeCard && nextCard
    ? clamp((scrollState.progress - 0.78) / 0.22, 0, 1)
    : 0;

  useEffect(() => {
    const update = () => {
      if (containerRef.current) setContainerHeight(containerRef.current.clientHeight);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  useEffect(() => {
    setScrollState({ progress: 0, atTop: true, atBottom: false });
    if (activeCard && contentRefs.current[activeCard]) {
      // Only reset scroll when there's no section to jump to — the card will
      // handle its own scroll via data-section otherwise.
      if (!focusedSection) {
        contentRefs.current[activeCard].scrollTo({ top: 0, behavior: 'auto' });
      }
    }
  }, [activeCard, focusedSection]);

  const registerScrollRef = useCallback((cardId) => (node) => {
    if (node) contentRefs.current[cardId] = node;
    else delete contentRefs.current[cardId];
  }, []);

  const moveToRelativeCard = useCallback((direction) => {
    if (!activeCard) return;
    const next = cards[activeIndex + direction];
    if (!next) return;
    onSelectCard(next.id);
  }, [activeCard, activeIndex, cards, onSelectCard]);

  const handleCardScroll = (cardId, event) => {
    if (cardId !== activeCard) return;
    setScrollState(getMetrics(event.currentTarget));
  };

  const handleCardWheel = (cardId, event) => {
    if (cardId !== activeCard) return;
    const metrics = getMetrics(event.currentTarget);
    if (metrics.atBottom && event.deltaY > 75 && nextCard) {
      event.preventDefault();
      moveToRelativeCard(1);
      return;
    }
    if (metrics.atTop && event.deltaY < -75 && prevCard) {
      event.preventDefault();
      moveToRelativeCard(-1);
    }
  };

  const handleTouchStart = (cardId, event) => {
    if (cardId !== activeCard) return;
    touchStartYRef.current = event.touches[0]?.clientY ?? null;
  };

  const handleTouchEnd = (cardId, event) => {
    if (cardId !== activeCard || touchStartYRef.current == null) return;
    const endY = event.changedTouches[0]?.clientY ?? touchStartYRef.current;
    const deltaY = touchStartYRef.current - endY;
    const metrics = getMetrics(event.currentTarget);
    if (metrics.atBottom && deltaY > 60 && nextCard) moveToRelativeCard(1);
    else if (metrics.atTop && deltaY < -60 && prevCard) moveToRelativeCard(-1);
    touchStartYRef.current = null;
  };

  const hiddenY = containerHeight + 36;
  const peekY = Math.max(containerHeight - 150, 112);

  const getCardState = (card, index) => {
    if (!activeCard) {
      return {
        y: index * 60,
        scale: 1 - index * 0.04,
        opacity: 1,
        zIndex: 20 - index,
      };
    }
    if (card.id === activeCard) {
      return { y: 0, scale: 1, opacity: 1, zIndex: 80 };
    }
    if (nextCard && card.id === nextCard.id) {
      return {
        y: hiddenY - (hiddenY - peekY) * peekProgress,
        scale: 0.96 + 0.02 * peekProgress,
        opacity: 0.15 + 0.82 * peekProgress,
        zIndex: 70,
      };
    }
    return { y: hiddenY, scale: 0.94, opacity: 0, zIndex: 10 - index };
  };

  // Click-outside-to-close, only when the user clicked the container itself
  // (not a child card). Prevents the peek-card click race.
  const handleContainerClick = (e) => {
    if (!activeCard) return;
    if (e.target === e.currentTarget) onClearActiveCard();
  };

  return (
    <div className="h-full px-3 pb-3 md:px-6 md:pb-6">
      <div
        ref={containerRef}
        className="relative h-full"
        onClick={handleContainerClick}
      >
        {!activeCard && (
          <div className="pointer-events-none absolute inset-x-0 top-1 z-0 px-5">
            <div className="rounded-full border border-subtle/10 bg-surface/50 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-faint backdrop-blur-sm">
              Lab · streaming agent · ⌘K for palette
            </div>
          </div>
        )}

        {cards.map((card, index) => (
          <LabCard
            key={card.id}
            card={card}
            state={getCardState(card, index)}
            isActive={card.id === activeCard}
            isHighlighted={card.id === highlightedCard}
            isPeekCard={nextCard?.id === card.id && peekProgress > 0}
            hasNextCard={card.id === activeCard && Boolean(nextCard)}
            focusedSection={card.id === activeCard ? focusedSection : null}
            highlightPulseKey={highlightPulseKey}
            onSelect={onSelectCard}
            onClose={onClearActiveCard}
            registerScrollRef={registerScrollRef}
            onCardScroll={handleCardScroll}
            onCardWheel={handleCardWheel}
            onTouchStart={handleTouchStart}
            onTouchEnd={handleTouchEnd}
          />
        ))}
      </div>
    </div>
  );
};

export default LabStack;
