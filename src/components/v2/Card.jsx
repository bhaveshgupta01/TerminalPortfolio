import React, { useEffect, useRef, useState } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { getMaterial } from './materials';
import { useV2Stack } from '../../context/V2StackContext';

/**
 * Single drawer card.
 *
 *   peek     — only label strip visible (top ~32px) above the active card
 *   active   — sitting at bottom; hover drives a soft 3D tilt + cursor-tracked
 *              highlight, both spring-smoothed via Framer Motion motion values
 *              (no React rerenders on mousemove → buttery)
 *   focused  — promoted detail in same slot, scrollable body, optional
 *              peek-progress hint bar at top/bottom edges. AI focus triggers
 *              a one-shot sheen sweep + buffering shimmer until first
 *              text_delta lands.
 *
 * Surface stack (bottom → top):
 *   gradient bg → sheen radial → top specular glaze → bevel lip
 *   → grain → brushed streaks → bottom shadow seal
 *   → cursor-tracked highlight (active hover)
 *   → buffering shimmer (AI active) → AI focus sheen sweep (one-shot)
 *   → content (header + body)
 */

const TRANS = '0.52s cubic-bezier(0.32, 0.72, 0, 1)';

// Hover springs — slow + heavy enough to feel like physical card, fast
// enough to track the cursor without lag.
const TILT_SPRING = { stiffness: 150, damping: 22, mass: 1.1 };
const LIFT_SPRING = { stiffness: 200, damping: 28, mass: 1 };

const PeekHintBar = ({ mat, label, progress, side }) => {
  const filled = Math.min(1, Math.max(0, progress));
  const visible = filled > 0;
  return (
    <div
      style={{
        position: 'absolute',
        left: 14,
        right: 14,
        [side]: 14,
        height: 30,
        borderRadius: 999,
        background: 'rgba(0,0,0,0.55)',
        border: `1px solid ${mat.mute}3a`,
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        opacity: visible ? 0.92 : 0.55,
        transform: visible ? 'translateY(0)' : `translateY(${side === 'bottom' ? 4 : -4}px)`,
        transition: 'opacity 200ms ease, transform 200ms ease',
        pointerEvents: 'none',
        zIndex: 6,
        backdropFilter: 'blur(8px)',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `linear-gradient(90deg, ${mat.accent}66 0%, ${mat.accent}33 100%)`,
          width: `${filled * 100}%`,
          transition: 'width 60ms linear',
        }}
      />
      <div
        style={{
          position: 'relative',
          flex: 1,
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 9,
          letterSpacing: 1.4,
          textTransform: 'uppercase',
          color: filled > 0.5 ? '#fff' : 'rgba(255,255,255,0.78)',
          textAlign: 'center',
          padding: '0 12px',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {side === 'bottom' ? '↓' : '↑'}
        &nbsp;&nbsp;
        {filled >= 1 ? 'release · ' : 'keep scrolling for · '}
        <span style={{ fontWeight: 600 }}>{label}</span>
      </div>
    </div>
  );
};

const CardHeader = ({ card, mat, focused, peekOnly }) => (
  <div
    style={{
      padding: focused
        ? '24px 30px 16px'
        : peekOnly
          ? '11px 24px 11px'
          : '18px 26px 16px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      gap: 16,
    }}
  >
    <div style={{ minWidth: 0, flex: 1 }}>
      <div
        style={{
          fontFamily: "'JetBrains Mono', ui-monospace, monospace",
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: 2.2,
          color: mat.tag,
          marginBottom: peekOnly ? 0 : 8,
          textTransform: 'uppercase',
          textShadow: '0 1px 0 rgba(0,0,0,0.35)',
        }}
      >
        {card.label}
      </div>
      {!peekOnly && (
        <>
          <div
            style={{
              fontFamily: "'Inter Tight', -apple-system, sans-serif",
              fontSize: focused ? 28 : 22,
              fontWeight: 600,
              letterSpacing: -0.5,
              color: mat.ink,
              lineHeight: 1.1,
              marginBottom: 4,
              textShadow: '0 1px 2px rgba(0,0,0,0.45)',
            }}
          >
            {card.title}
          </div>
          {card.subtitle && (
            <div
              style={{
                fontFamily: "'Inter Tight', sans-serif",
                fontSize: 13,
                color: mat.ink,
                opacity: 0.72,
                fontWeight: 400,
              }}
            >
              {card.subtitle}
            </div>
          )}
        </>
      )}
    </div>
  </div>
);

const Card = React.forwardRef(({
  card,
  mode = 'peek',
  width,
  height,
  bodyRef,
  onBodyScroll,
  onBodyWheel,
  peekHint,
}, outerRef) => {
  const [hover, setHover] = useState(false);
  const [revealKey, setRevealKey] = useState(0);
  const innerRef = useRef(null);
  const { bufferingCardId, focusRevealKey, activeSource } = useV2Stack();
  const isBuffering = bufferingCardId === card.id;

  const mat = getMaterial(card.material);
  const peekOnly = mode === 'peek';
  const focused = mode === 'focused';
  const isActive = mode === 'active';
  const Body = card.render;

  // ── Spring-smoothed cursor parallax (no React state — no rerenders on mousemove) ──
  const mx = useMotionValue(0);          // -1 (left) … 1 (right)
  const my = useMotionValue(0);          // -1 (top)  … 1 (bottom)
  const sx = useSpring(mx, TILT_SPRING);
  const sy = useSpring(my, TILT_SPRING);
  // Tilt amplitude depends on mode — full 6° for active card, gentler 3°
  // for peek tabs so hovering a stack tab doesn't feel "loud".
  const tiltAmp = mode === 'active' ? 6 : mode === 'peek' ? 3 : 0;
  const rotateY = useTransform(sx, [-1, 1], [-tiltAmp, tiltAmp]);
  const rotateX = useTransform(sy, [-1, 1], [tiltAmp, -tiltAmp]);
  // Cursor highlight tracks the cursor smoothly via the same springs.
  const hxPct = useTransform(sx, [-1, 1], ['10%', '90%']);
  const hyPct = useTransform(sy, [-1, 1], ['10%', '90%']);
  // Composed background string for the cursor-tracked highlight.
  // MUST be declared at the top of the component (Rules of Hooks) — calling
  // useTransform inside conditional JSX would crash on every isActive flip.
  const highlightBg = useTransform(
    [hxPct, hyPct],
    ([x, y]) => `radial-gradient(360px 240px at ${x} ${y}, rgba(255,255,255,0.18), transparent 70%)`,
  );

  // Hover lift (animated separately so it doesn't fight the tilt).
  // Active cards lift -8px on hover, peek tabs lift only -3px (just enough
  // to show "I'm being acknowledged"), focused doesn't lift at all.
  const lift = useMotionValue(0);
  const liftSpring = useSpring(lift, LIFT_SPRING);

  // When the agent triggers focus on this card, fire a one-shot sheen sweep.
  useEffect(() => {
    if (mode === 'focused' && activeSource === 'ai') {
      setRevealKey((k) => k + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusRevealKey]);

  useEffect(() => {
    if (focused) lift.set(0);
    else if (hover && isActive) lift.set(-8);
    else if (hover) lift.set(-3); // gentle nudge for peek tabs
    else lift.set(0);
  }, [focused, hover, isActive, lift]);

  const handleMouseMove = (e) => {
    if (focused) return; // focused card doesn't tilt — the body needs stable scroll
    const r = e.currentTarget.getBoundingClientRect();
    mx.set(((e.clientX - r.left) / r.width) * 2 - 1);
    my.set(((e.clientY - r.top) / r.height) * 2 - 1);
  };

  const handleMouseLeave = () => {
    setHover(false);
    mx.set(0);
    my.set(0);
  };

  // Shadow strength varies by mode and hover — but transitions slowly so it
  // doesn't read as "snappy".
  const baseShadow = focused
    ? `0 36px 88px rgba(0,0,0,0.7), 0 6px 16px rgba(0,0,0,0.45), 0 0 60px ${mat.accent}1c`
    : isActive
      ? hover
        ? `0 22px 44px rgba(0,0,0,0.62), 0 4px 10px rgba(0,0,0,0.5), 0 0 56px ${mat.accent}3a`
        : `0 14px 28px rgba(0,0,0,0.55), 0 2px 6px rgba(0,0,0,0.5), 0 0 36px ${mat.accent}1c`
      : '0 8px 18px rgba(0,0,0,0.45), 0 1px 3px rgba(0,0,0,0.5)';

  return (
    <motion.div
      ref={(node) => {
        innerRef.current = node;
        if (typeof outerRef === 'function') outerRef(node);
        else if (outerRef) outerRef.current = node;
      }}
      onMouseEnter={() => !focused && setHover(true)}
      onMouseLeave={handleMouseLeave}
      onMouseMove={handleMouseMove}
      role={focused ? 'region' : 'button'}
      tabIndex={!focused ? 0 : -1}
      aria-label={`${card.title} card`}
      style={{
        position: 'absolute',
        width,
        height,
        borderRadius: 22,
        cursor: focused ? 'default' : 'pointer',
        overflow: 'hidden',
        background: mat.bg,
        boxShadow: `
          0 1px 0 rgba(255,255,255,0.18) inset,
          0 -1px 0 rgba(0,0,0,0.6) inset,
          0 0 0 1px ${mat.accent}28,
          ${baseShadow}
        `,
        border: `1px solid ${mat.accent}30`,
        // 3D tilt — applied to all non-focused cards. Tilt amplitude varies
        // by mode (set above): full for active, gentle for peek tabs, zero
        // for focused. Spring values default back to 0 on mouseleave.
        rotateX: focused ? 0 : rotateX,
        rotateY: focused ? 0 : rotateY,
        y: liftSpring,
        transformStyle: 'preserve-3d',
        transformPerspective: 1200,
        // CSS transition for shadow + the box itself when not under spring control.
        transition: `box-shadow ${TRANS}, border-color ${TRANS}`,
        willChange: 'transform',
      }}
    >
      {/* sheen radial — anisotropic top half, the "rolled metal" feel */}
      <div style={{ position: 'absolute', inset: 0, background: mat.sheen, pointerEvents: 'none' }} />
      {/* top specular glaze (sharp top-left) */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: mat.specular,
          pointerEvents: 'none',
          opacity: hover ? 1.15 : 1,
          transition: `opacity ${TRANS}`,
        }}
      />
      {/* bevel — bright lip along the very top edge */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 18,
          background: mat.bevel,
          pointerEvents: 'none',
          opacity: 0.95,
        }}
      />
      {/* base shadow seal */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 60,
          background: mat.base,
          pointerEvents: 'none',
        }}
      />
      {/* grain */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence baseFrequency='0.85' numOctaves='2'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>\")",
          mixBlendMode: 'overlay',
          opacity: 0.50,
        }}
      />
      {/* brushed-metal vertical streaks */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background:
            'repeating-linear-gradient(90deg, rgba(255,255,255,0.030) 0px, rgba(255,255,255,0) 1.5px, rgba(0,0,0,0.030) 3px, rgba(0,0,0,0) 4.5px)',
          opacity: 0.62,
        }}
      />

      {/* Cursor-tracked highlight — spring-smoothed via motion values, no
          React state churn so this is buttery on every mousemove. The
          underlying useTransform is created at the top of the component
          unconditionally (Rules of Hooks). Always rendered; opacity
          toggles based on isActive + hover so the hook count stays stable. */}
      <motion.div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background: highlightBg,
          mixBlendMode: 'screen',
          opacity: isActive && hover ? 1 : 0,
          transition: `opacity ${TRANS}`,
        }}
      />

      {/* AI focus reveal sheen sweep — only fires when the AGENT focuses */}
      {focused && revealKey > 0 && (
        <div key={`reveal-${revealKey}`} className="v2-sheen v2-sheen-once" />
      )}

      <div style={{ position: 'relative', height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardHeader card={card} mat={mat} focused={focused} peekOnly={peekOnly} />

        {focused && (
          <div
            ref={bodyRef}
            onScroll={onBodyScroll}
            onWheel={onBodyWheel}
            className="v2-card-body"
            style={{
              flex: 1,
              overflowY: 'auto',
              overflowX: 'hidden',
              paddingBottom: 60,
              overscrollBehavior: 'contain',
              position: 'relative',
            }}
          >
            <Body mat={mat} />
            {isBuffering && <div className="v2-buffer-shimmer" aria-hidden />}
          </div>
        )}

        {focused && peekHint?.prev && (
          <PeekHintBar mat={mat} side="top" label={peekHint.prev.label} progress={peekHint.prevProgress} />
        )}
        {focused && peekHint?.next && (
          <PeekHintBar mat={mat} side="bottom" label={peekHint.next.label} progress={peekHint.nextProgress} />
        )}
      </div>
    </motion.div>
  );
});

Card.displayName = 'Card';

export default Card;
