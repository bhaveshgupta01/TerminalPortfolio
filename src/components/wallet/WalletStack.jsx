import React, { useEffect, useRef, useState } from 'react';
import WalletCard from './WalletCard';

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const getMetrics = (node) => {
  const maxScrollTop = Math.max(node.scrollHeight - node.clientHeight, 0);
  const progress = maxScrollTop > 0 ? node.scrollTop / maxScrollTop : 0;

  return {
    atTop: node.scrollTop <= 4,
    atBottom: node.scrollTop >= maxScrollTop - 4,
    progress,
  };
};

const WalletStack = ({
  cards,
  activeCard,
  highlightedCard,
  onSelectCard,
  onClearActiveCard,
}) => {
  const containerRef = useRef(null);
  const contentRefs = useRef({});
  const touchStartYRef = useRef(null);
  const [containerHeight, setContainerHeight] = useState(480);
  const [scrollState, setScrollState] = useState({
    progress: 0,
    atTop: true,
    atBottom: false,
  });

  const activeIndex = cards.findIndex((card) => card.id === activeCard);
  const nextCard = activeIndex >= 0 ? cards[activeIndex + 1] : null;
  const prevCard = activeIndex > 0 ? cards[activeIndex - 1] : null;
  const peekProgress = activeCard && nextCard
    ? clamp((scrollState.progress - 0.78) / 0.22, 0, 1)
    : 0;

  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) setContainerHeight(containerRef.current.clientHeight);
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  useEffect(() => {
    setScrollState({ progress: 0, atTop: true, atBottom: false });
    if (activeCard && contentRefs.current[activeCard]) {
      contentRefs.current[activeCard].scrollTo({ top: 0, behavior: 'auto' });
    }
  }, [activeCard]);

  const registerScrollRef = (cardId) => (node) => {
    if (node) {
      contentRefs.current[cardId] = node;
    } else {
      delete contentRefs.current[cardId];
    }
  };

  const moveToRelativeCard = (direction) => {
    if (!activeCard) return;
    const nextIndex = activeIndex + direction;
    const nextTarget = cards[nextIndex];
    if (!nextTarget) return;
    onSelectCard(nextTarget.id);
  };

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

    if (metrics.atBottom && deltaY > 60 && nextCard) {
      moveToRelativeCard(1);
    } else if (metrics.atTop && deltaY < -60 && prevCard) {
      moveToRelativeCard(-1);
    }

    touchStartYRef.current = null;
  };

  const hiddenY = containerHeight + 36;
  const peekY = Math.max(containerHeight - 150, 112);

  const getCardState = (card, index) => {
    if (!activeCard) {
      return {
        y: index * 60,
        scale: 1 - (index * 0.04),
        opacity: 1,
        zIndex: 20 - index,
      };
    }

    if (card.id === activeCard) {
      return {
        y: 0,
        scale: 1,
        opacity: 1,
        zIndex: 80,
      };
    }

    if (nextCard && card.id === nextCard.id) {
      return {
        y: hiddenY - ((hiddenY - peekY) * peekProgress),
        scale: 0.96 + (0.02 * peekProgress),
        opacity: 0.15 + (0.82 * peekProgress),
        zIndex: 70,
      };
    }

    return {
      y: hiddenY,
      scale: 0.94,
      opacity: 0,
      zIndex: 10 - index,
    };
  };

  return (
    <div className="h-full px-3 pb-3 md:px-6 md:pb-6">
      <div
        ref={containerRef}
        className="relative h-full"
        onClick={activeCard ? onClearActiveCard : undefined}
      >
        {!activeCard && (
          <div className="pointer-events-none absolute inset-x-0 top-1 z-0 px-5">
            <div className="rounded-full border border-subtle/10 bg-surface/50 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-faint backdrop-blur-sm">
              Apple Wallet prototype
            </div>
          </div>
        )}

        {cards.map((card, index) => (
          <WalletCard
            key={card.id}
            card={card}
            state={getCardState(card, index)}
            isActive={card.id === activeCard}
            isHighlighted={card.id === highlightedCard}
            isPeekCard={nextCard?.id === card.id && peekProgress > 0}
            hasNextCard={card.id === activeCard && Boolean(nextCard)}
            onSelect={onSelectCard}
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

export default WalletStack;
