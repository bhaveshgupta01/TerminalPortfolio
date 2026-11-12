import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useV2Stack } from '../../context/V2StackContext';
import Card from './Card';
import { getDeck } from './decks';

/**
 * Drawer-style card stack.
 *
 * Layout (no 3D, just `top` / `height` animations):
 *   - all cards absolutely positioned inside a centred wrapper at the
 *     bottom of the available height
 *   - card behind the active one shows only its label strip (PEEK_TAB px)
 *     above the active card's top edge — its body extends DOWN behind the
 *     active card, invisible
 *   - cards already passed sink slightly under the active card
 *   - active card sits at bottom=0, height = ACTIVE_H
 *   - focused (open) card grows to FOCUS_H, dim overlay covers the rest
 *
 * Scroll behaviour:
 *   - in stack mode (no focus) — one wheel/touch/key step = one card
 *     advance, with a 600ms cooldown so trackpad inertia can't chain
 *   - in focused mode — internal body scroll first; at content bottom
 *     keep scrolling and the next card peeks (peekOverflow grows); past
 *     COMMIT_PX, advance focus to that next card. Symmetric upward.
 */

const PEEK_TAB = 34;          // tab strip height for peeking cards
const ACTIVE_H_DESKTOP = 220;  // closed/active card height (desktop)
const ACTIVE_H_MOBILE_MIN = 160; // min closed card height on tight mobile zones
const FOCUS_H_MAX = 640;       // max focused card height
const FOCUS_H_MIN_OFFSET = 96; // min space to leave above + below focused card
const FOCUS_H_MIN_FLOOR = 240; // absolute floor — below this, focused mode would be unusable

// Passed-card slivers below the active card. When a user lands deep in the
// deck (e.g. /v2/skills, /v2/contact) these tiny edges signal "more cards
// live below" so they don't mistake the top 4 peek tabs for the entire deck.
const PASSED_PEEK = 6;         // px sliver each visible passed card shows
const PASSED_PEEK_MAX = 4;     // max number of passed slivers (older cards fully hidden)

const STEP_COOLDOWN = 520;     // ms between accepted scroll steps
const PEEK_UNLOCK = 24;        // wheel deltaY past content edge before peek begins
const COMMIT_PX = 120;         // wheel deltaY past peek-unlock to commit
const TRANSITION = '0.55s cubic-bezier(0.32, 0.72, 0, 1)'; // shared transition (slightly slower for play)

const CardStack = () => {
  const {
    activeCardId,
    openMode,
    activeDeckId,
    focusedSection,
    highlightPulseKey,
    navigate,
    agentStage,
  } = useV2Stack();

  // We need the deck cards. Decks are static config; lazily import to avoid a
  // ts-shaped circular dep with V2Home if it ever happens.
  // (Static import is fine — kept as require-style for clarity.)
  const cards = useDeckCards(activeDeckId);

  const wrapperRef = useRef(null);
  const bodyRef = useRef(null);
  const stepLockedUntil = useRef(0);

  const [stackH, setStackH] = useState(640);
  const [stackW, setStackW] = useState(1000);
  const [overflow, setOverflow] = useState({ next: 0, prev: 0 });

  // Responsive card width — bumped for desktop so the cards actually fill
  // the canvas instead of looking lonely on a 1920px screen.
  const cardWidth = useMemo(() => {
    if (stackW < 480) return Math.max(300, Math.min(stackW * 0.92, 400));
    if (stackW < 900) return 480;
    if (stackW < 1280) return 580;
    if (stackW < 1700) return 660;
    return 720;
  }, [stackW]);

  // Adaptive active-card height: shrink on tight mobile zones (e.g. when the
  // chat bar is expanded and the stack zone is < 360px tall) so the closed
  // card and a peek tab can both fit.
  const ACTIVE_H = stackH < 360
    ? Math.max(ACTIVE_H_MOBILE_MIN, stackH - 110) // leave room for one peek tab + breathing
    : ACTIVE_H_DESKTOP;

  // Focused-card height: fill most of available space, capped.
  // Floor at FOCUS_H_MIN_FLOOR so the focused body stays usable; cap at the
  // stack zone height so the card can never overflow its container.
  const focusH = Math.max(
    FOCUS_H_MIN_FLOOR,
    Math.min(FOCUS_H_MAX, stackH - FOCUS_H_MIN_OFFSET),
  );

  // How many upcoming peek tabs fit above the active card?
  // Leave ~120px breathing room at the very top of the stack zone.
  // On tight zones (mobile + expanded chat), force at least 1 tab visible.
  const visibleTabs = Math.max(
    stackH < 360 ? 1 : 2,
    Math.floor((stackH - ACTIVE_H - 120) / PEEK_TAB),
  );

  // Find current index. Default to 0 if the URL doesn't name one.
  const activeIndex = useMemo(() => {
    if (!activeCardId) return 0;
    const i = cards.findIndex((c) => c.id === activeCardId);
    return i < 0 ? 0 : i;
  }, [activeCardId, cards]);

  const focused = openMode;

  // ── Virtual cycling: while the AI is thinking / searching, visually
  // flip through the deck so the user can see the agent "looking through
  // the wallet". When ui_command lands (or stage settles), this clears
  // and the activeIndex from the URL takes over.
  const [virtualIdx, setVirtualIdx] = useState(null);
  const cycleTimer = useRef(null);
  const isSearching = !focused && (
    agentStage === 'thinking' ||
    agentStage === 'searching_profile' ||
    agentStage === 'searching_web'
  );

  useEffect(() => {
    if (cycleTimer.current) {
      clearTimeout(cycleTimer.current);
      cycleTimer.current = null;
    }
    if (!isSearching) {
      setVirtualIdx(null);
      return undefined;
    }
    let i = activeIndex;
    let count = 0;
    const tick = () => {
      i = (i + 1) % cards.length;
      setVirtualIdx(i);
      count += 1;
      // Fan feel — steady ~280ms per card so the eye can track each card
      // briefly lifting + dropping like the agent is rifling through.
      // Slows further as we go (converging on the answer).
      const delay = 240 + Math.min(count * 10, 120);
      cycleTimer.current = setTimeout(tick, delay);
    };
    cycleTimer.current = setTimeout(tick, 160);
    return () => {
      if (cycleTimer.current) {
        clearTimeout(cycleTimer.current);
        cycleTimer.current = null;
      }
    };
  }, [isSearching, activeIndex, cards.length]);

  // Use virtualIdx for layout while cycling; activeIndex otherwise.
  const displayedIndex = virtualIdx != null ? virtualIdx : activeIndex;

  // Reset overflow trackers whenever the focused card changes.
  useEffect(() => {
    setOverflow({ next: 0, prev: 0 });
    if (focused && bodyRef.current && !focusedSection) {
      bodyRef.current.scrollTo({ top: 0, behavior: 'auto' });
    }
  }, [activeCardId, focused, focusedSection]);

  // Subsection scroll + pulse ring in the focused body.
  useEffect(() => {
    if (!focused || !focusedSection) return;
    const body = bodyRef.current;
    if (!body) return;
    const el = body.querySelector(`[data-section="${focusedSection}"]`);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    el.classList.remove('v2-section-ring');
    void el.offsetWidth; // restart keyframe
    el.classList.add('v2-section-ring');
  }, [focused, focusedSection, highlightPulseKey, activeCardId]);

  // ── viewport size ────────────────────────────────────────────────────────
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setStackH(e.contentRect.height);
        setStackW(e.contentRect.width);
      }
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setStackH(r.height);
    setStackW(r.width);
    return () => ro.disconnect();
  }, []);

  // ── stepping helpers ─────────────────────────────────────────────────────
  const stepStack = useCallback((delta) => {
    const now = performance.now();
    if (now < stepLockedUntil.current) return;
    stepLockedUntil.current = now + STEP_COOLDOWN;
    const next = cards[activeIndex + delta];
    if (!next) return;
    navigate(
      { deckId: activeDeckId, cardId: next.id, open: focused },
      { source: 'user' },
    );
  }, [activeDeckId, activeIndex, cards, focused, navigate]);

  // ── stack-mode wheel / touch / key ───────────────────────────────────────
  useEffect(() => {
    if (focused) return; // focused mode handles its own scroll inside the body
    const el = wrapperRef.current;
    if (!el) return;

    const onWheel = (e) => {
      if (Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;
      // Always prevent — even small deltaY would otherwise bubble to body
      // and cause that "page bounces a little" feel the user reported.
      e.preventDefault();
      if (Math.abs(e.deltaY) < 12) return; // deadzone for trackpad jitter
      // Reversed: scrolling UP advances the stack (next card lives ABOVE the
      // active in our drawer, so dragging up to "pull it down" feels right).
      stepStack(e.deltaY > 0 ? -1 : 1);
    };

    let touchY = null;
    const onTouchStart = (ev) => { touchY = ev.touches[0]?.clientY ?? null; };
    const onTouchEnd = (ev) => {
      if (touchY == null) return;
      const dy = touchY - (ev.changedTouches[0]?.clientY ?? touchY);
      if (Math.abs(dy) < 30) { touchY = null; return; }
      // Reversed: swipe DOWN (dy < 0) → next card; swipe UP (dy > 0) → previous.
      stepStack(dy > 0 ? -1 : 1);
      touchY = null;
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('touchstart', onTouchStart, { passive: true });
    el.addEventListener('touchend', onTouchEnd, { passive: true });
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchend', onTouchEnd);
    };
  }, [focused, stepStack]);

  // ── focused-mode wheel: peek then commit ─────────────────────────────────
  // Keep peek-and-commit but follow the same reversed direction as stack
  // mode: scrolling UP at body top = previous; scrolling DOWN at body
  // bottom = next. (Inside the body, normal scroll-direction conventions
  // already apply because the browser's native scroll handles content.)
  const handleBodyWheel = useCallback((e) => {
    const body = bodyRef.current;
    if (!body) return;

    const max = Math.max(body.scrollHeight - body.clientHeight, 0);
    const atBottom = max <= 0 || body.scrollTop >= max - 2;
    const atTop = body.scrollTop <= 2;

    // overflow.next  → growing means we're committing to the NEXT card
    // overflow.prev  → growing means we're committing to the PREVIOUS card
    if (atBottom && e.deltaY > 0) {
      e.preventDefault();
      const next = overflow.next + e.deltaY;
      if (next >= PEEK_UNLOCK + COMMIT_PX) {
        setOverflow({ next: 0, prev: 0 });
        const target = cards[activeIndex + 1];
        if (target) navigate({ deckId: activeDeckId, cardId: target.id, open: true }, { source: 'user' });
      } else {
        setOverflow({ next, prev: 0 });
      }
    } else if (atTop && e.deltaY < 0) {
      e.preventDefault();
      const prev = overflow.prev + -e.deltaY;
      if (prev >= PEEK_UNLOCK + COMMIT_PX) {
        setOverflow({ next: 0, prev: 0 });
        const target = cards[activeIndex - 1];
        if (target) navigate({ deckId: activeDeckId, cardId: target.id, open: true }, { source: 'user' });
      } else {
        setOverflow({ next: 0, prev });
      }
    } else if (overflow.next !== 0 || overflow.prev !== 0) {
      setOverflow({ next: 0, prev: 0 });
    }
  }, [activeDeckId, activeIndex, cards, navigate, overflow]);

  // peek progress 0..1 — used by the hint bar at bottom of the focused card
  const peekNextProgress = Math.min(1, Math.max(0, (overflow.next - PEEK_UNLOCK) / COMMIT_PX));
  const peekPrevProgress = Math.min(1, Math.max(0, (overflow.prev - PEEK_UNLOCK) / COMMIT_PX));

  // ── keyboard ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      const tag = (document.activeElement?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      if (e.key === 'Escape' && focused) {
        navigate({ deckId: activeDeckId, cardId: activeCardId, open: false }, { source: 'user' });
        return;
      }
      // Match the reversed wheel: ArrowUp = next (pulls it down into active),
      // ArrowDown = previous. ArrowLeft / ArrowRight follow the same model.
      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault();
        const next = cards[activeIndex + 1];
        if (next) navigate({ deckId: activeDeckId, cardId: next.id, open: focused }, { source: 'user' });
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown') {
        e.preventDefault();
        const prev = cards[activeIndex - 1];
        if (prev) navigate({ deckId: activeDeckId, cardId: prev.id, open: focused }, { source: 'user' });
      } else if (e.key === 'Enter' && !focused) {
        e.preventDefault();
        const here = cards[activeIndex];
        if (here) navigate({ deckId: activeDeckId, cardId: here.id, open: true }, { source: 'user' });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [activeCardId, activeDeckId, activeIndex, cards, focused, navigate]);

  // ── per-card layout calculation ──────────────────────────────────────────
  //
  // Drawer model — every card is anchored at the bottom of the wrapper via
  // `bottom`. Cards are 220px tall (closed) or focusH tall (focused).
  //
  // Stack mode (no focus):
  //   active   → bottom: 0,            height: 220, z: 100, mode: 'active'
  //   upcoming → bottom: d*PEEK_TAB,    height: 220, z: 100-d (BEHIND active),
  //              mode: 'peek'.  Card's body extends down behind the active
  //              card; only its top label strip pokes above.
  //   passed   → bottom: -d*3,          height: 220, z: 50-d, sunk under
  //              active, only tiny edge visible
  //
  // Focused mode:
  //   focused card → bottom: 24, height: focusH, z: 1000, mode: 'focused'
  //   everyone else → bunched at bottom under the dim overlay
  //                   (bottom: i*3, z: 30-d, low opacity, mode: 'peek')
  //
  // How much room to reserve below the active card for passed-card slivers.
  // Caps at PASSED_PEEK_MAX * PASSED_PEEK so the active card never floats
  // too high (e.g. on /v2/contact with 9 passed cards we still show only 4).
  const passedSlot = Math.min(displayedIndex, PASSED_PEEK_MAX) * PASSED_PEEK;

  const cardLayouts = useMemo(() => {
    return cards.map((card, i) => {
      const d = i - displayedIndex;
      const absD = Math.abs(d);

      // ── FOCUSED MODE ─────────────────────────────────────────────
      // The focused card sits center; every other card bunches at the
      // bottom edge of the wrapper as a tight visible stack ("I'm still
      // here") instead of vanishing.
      if (focused) {
        if (i === displayedIndex) {
          return { card, bottom: 24, height: focusH, z: 1000, mode: 'focused', opacity: 1 };
        }
        // Bunched stack: each card adds a tiny stagger (4px each for the
        // first 5, 1.5px each beyond), opacity floor at 0.18.
        const order = absD;
        const close = order <= 5;
        const bottomOffset = close ? 6 + order * 4 : 26 + (order - 5) * 1.5;
        const opacity = close
          ? Math.max(0.22, 0.55 - order * 0.07)
          : Math.max(0.16, 0.22 - (order - 5) * 0.015);
        return {
          card,
          bottom: bottomOffset,
          height: ACTIVE_H,
          z: 60 - order,
          mode: 'peek',
          opacity,
        };
      }

      // ── STACK MODE: ACTIVE ──────────────────────────────────────
      // Lifted by passedSlot so the passed-card slivers below have room
      // to peek out. passedSlot is 0 when nothing is passed (home view).
      if (i === displayedIndex) {
        return { card, bottom: passedSlot, height: ACTIVE_H, z: 100, mode: 'active', opacity: 1 };
      }

      // ── STACK MODE: UPCOMING (cards behind/above active) ────────
      if (d > 0) {
        // Within visible peek tab cap: regular stagger.
        // Beyond cap: bunch tightly at the top edge (+2px each) so the
        // user always sees "more cards back here" instead of nothing.
        // +passedSlot keeps the peek-tab geometry relative to the active
        // card's top edge (which is now also lifted).
        const clamped = Math.min(d, visibleTabs);
        const extra = Math.max(0, d - visibleTabs);
        return {
          card,
          bottom: clamped * PEEK_TAB + extra * 2 + passedSlot,
          height: ACTIVE_H,
          z: 100 - d,
          mode: 'peek',
          opacity: extra > 0 ? Math.max(0.35, 0.85 - extra * 0.06) : 1,
        };
      }

      // ── STACK MODE: PASSED (cards user has scrolled past) ───────
      // Each visible passed card sits PASSED_PEEK px lower than the next,
      // forming a tight deck-of-cards sliver below the active card.
      // Z-index falls with distance so the most-recent passed draws on top.
      const passedD = -d;
      const visible = passedD <= PASSED_PEEK_MAX;
      const slot = Math.min(passedD, PASSED_PEEK_MAX);
      return {
        card,
        // Most recent (passedD=1) sits just below active; oldest (passedD=PASSED_PEEK_MAX)
        // sits at the wrapper floor.
        bottom: visible ? passedSlot - slot * PASSED_PEEK : -ACTIVE_H,
        height: ACTIVE_H,
        z: 50 - passedD,
        mode: 'peek',
        // Fade slightly with depth so the slivers feel like a tonal deck.
        opacity: visible ? Math.max(0.6, 0.92 - passedD * 0.07) : 0,
      };
    });
  }, [cards, focused, displayedIndex, focusH, visibleTabs, passedSlot]);

  return (
    <div
      ref={wrapperRef}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {/* ground glow */}
      <div
        aria-hidden
        style={{
          position: 'absolute', left: '50%', bottom: 0,
          transform: 'translateX(-50%)',
          width: '80%', height: '32%',
          background: 'radial-gradient(ellipse at 50% 100%, var(--v2-floor-shadow), transparent 60%)',
          pointerEvents: 'none',
        }}
      />

      {/* Dim overlay when focused — click closes.
          Gradient fades to ~0.18 alpha at the very bottom so the bunched
          stack of "I'm still here" cards peeks through faintly without
          being washed out. */}
      {focused && (
        <div
          onClick={() => navigate(
            { deckId: activeDeckId, cardId: activeCardId, open: false },
            { source: 'user' },
          )}
          style={{
            position: 'absolute',
            inset: 0,
            background: 'var(--v2-dim-overlay)',
            backdropFilter: 'blur(3px)',
            zIndex: 500,
            cursor: 'pointer',
          }}
        />
      )}

      {/* card column — centered horizontally, anchored to bottom */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          bottom: 0,
          transform: 'translateX(-50%)',
          width: cardWidth,
          height: '100%',
          zIndex: 600,
        }}
      >
        {cardLayouts.map(({ card, bottom, height, z, mode, opacity }) => {
          const isActive = mode === 'active';
          // While AI is auto-cycling, use a smoother per-card transition
          // (~340ms with a slight overshoot) so the deck flows like a fan
          // riffle, not a series of snaps.
          const cardTransition = isSearching
            ? 'bottom 0.34s cubic-bezier(0.34, 1.16, 0.64, 1), opacity 0.32s ease, transform 0.34s cubic-bezier(0.34, 1.16, 0.64, 1)'
            : `bottom ${TRANSITION}, height ${TRANSITION}, opacity 0.32s ease, transform ${TRANSITION}`;
          // Fan lift: while cycling, the currently-shown card lifts a
          // touch and tilts ~1.5° so the riffle animation reads as the
          // agent flipping cards.
          const fanLift = isSearching && isActive ? 'translateY(-6px) rotate(-1.4deg)' : 'translateY(0) rotate(0)';
          return (
            <div
              key={card.id}
              data-card-id={card.id}
              style={{
                position: 'absolute',
                left: 0,
                bottom,
                width: cardWidth,
                height,
                zIndex: z,
                opacity,
                transition: cardTransition,
                transform: isSearching ? fanLift : undefined,
              }}
              onClick={(e) => {
                if (mode === 'focused' || isSearching) return;
                e.stopPropagation();
                navigate(
                  { deckId: activeDeckId, cardId: card.id, open: isActive },
                  { source: 'user' },
                );
              }}
            >
              <Card
                card={card}
                mode={mode}
                width={cardWidth}
                height={height}
                bodyRef={mode === 'focused' ? bodyRef : undefined}
                onBodyWheel={mode === 'focused' ? handleBodyWheel : undefined}
                peekHint={mode === 'focused' ? {
                  next: cards[activeIndex + 1] || null,
                  prev: cards[activeIndex - 1] || null,
                  nextProgress: peekNextProgress,
                  prevProgress: peekPrevProgress,
                } : null}
              />
            </div>
          );
        })}
      </div>

      {/* focused state hint — bottom centre */}
      {focused && (
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: '50%',
            transform: 'translateX(-50%)',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            letterSpacing: 2,
            color: 'var(--v2-overlay-text)',
            textTransform: 'uppercase',
            zIndex: 1100,
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          scroll for next · click outside or esc to close
        </div>
      )}
    </div>
  );
};

/* Encapsulates deck lookup so future sub-decks can plug in here without
 * touching the rest of the stack code. */
const useDeckCards = (deckId) => getDeck(deckId).cards;

export default CardStack;
