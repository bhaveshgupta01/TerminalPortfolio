import React, { useEffect, useRef, useState } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

/**
 * Global cursor companion — a custom-drawn cartoon panda that lazily follows
 * the pointer across every route. Hand-built as inline SVG so it doesn't
 * duplicate the photographic /panda.png mascot used in chat avatars.
 *
 * After ~4 seconds of pointer inactivity it drifts to the bottom-right
 * corner and "sits" with a gentle breathing loop + slight tilt. Wakes the
 * moment the pointer moves again. Blinks every few seconds for life.
 *
 * Constraints honoured:
 *   - position: fixed + pointer-events: none → never intercepts clicks
 *   - hidden on touch / coarse-pointer devices via matchMedia('(hover: hover)')
 *   - hidden on prefers-reduced-motion
 */

const SIZE = 38;            // cartoon details need a touch more room
const REST_PAD = 28;        // px from viewport edge when sitting
const IDLE_MS = 4000;       // ms of stillness before the panda goes to sit
const CURSOR_OFFSET = 18;   // panda sits slightly below-right of the cursor

// Palette for the panda — kept fixed across themes because a panda IS
// black-and-white; the theme-shifting happens in the drop shadow only.
const INK = '#1a1a1a';
const FUR = '#f8f8f6';     // warm off-white, easier on the eye than pure white

const PandaSvg = ({ blinking }) => (
  <svg
    viewBox="0 0 100 100"
    width="100%"
    height="100%"
    aria-hidden
    style={{ display: 'block', overflow: 'visible' }}
  >
    {/* Arms (behind body, sticking out the sides) */}
    <ellipse cx="22" cy="72" rx="9" ry="6" fill={INK} transform="rotate(-22 22 72)" />
    <ellipse cx="78" cy="72" rx="9" ry="6" fill={INK} transform="rotate(22 78 72)" />

    {/* Body */}
    <ellipse cx="50" cy="78" rx="22" ry="16" fill={FUR} stroke={INK} strokeWidth="1.5" />

    {/* Feet — small black ovals peeking out at the bottom */}
    <ellipse cx="38" cy="93" rx="9" ry="5" fill={INK} />
    <ellipse cx="62" cy="93" rx="9" ry="5" fill={INK} />

    {/* Ears (drawn before head so head clips the inner edge) */}
    <circle cx="28" cy="16" r="11" fill={INK} />
    <circle cx="72" cy="16" r="11" fill={INK} />
    {/* Inner ear hint — subtle warm tone */}
    <circle cx="28" cy="19" r="4.5" fill="#3a2424" opacity="0.7" />
    <circle cx="72" cy="19" r="4.5" fill="#3a2424" opacity="0.7" />

    {/* Head */}
    <circle cx="50" cy="40" r="28" fill={FUR} stroke={INK} strokeWidth="1.5" />

    {/* Eye patches — rotated ovals leaning toward the nose */}
    <ellipse
      cx="37" cy="40" rx="7" ry="9.5"
      fill={INK}
      transform="rotate(-18 37 40)"
    />
    <ellipse
      cx="63" cy="40" rx="7" ry="9.5"
      fill={INK}
      transform="rotate(18 63 40)"
    />

    {/* Eye whites — collapse vertically during blink */}
    <ellipse
      cx="39" cy="41"
      rx={blinking ? 2.5 : 2.6}
      ry={blinking ? 0.3 : 2.6}
      fill={FUR}
      style={{ transition: 'ry 90ms ease-in' }}
    />
    <ellipse
      cx="61" cy="41"
      rx={blinking ? 2.5 : 2.6}
      ry={blinking ? 0.3 : 2.6}
      fill={FUR}
      style={{ transition: 'ry 90ms ease-in' }}
    />

    {/* Pupils — hidden during blink */}
    {!blinking && (
      <>
        <circle cx="40" cy="41.5" r="1.5" fill={INK} />
        <circle cx="60" cy="41.5" r="1.5" fill={INK} />
        {/* sparkle */}
        <circle cx="40.6" cy="40.9" r="0.55" fill={FUR} />
        <circle cx="60.6" cy="40.9" r="0.55" fill={FUR} />
      </>
    )}

    {/* Nose */}
    <ellipse cx="50" cy="51" rx="3" ry="2.2" fill={INK} />

    {/* Smile */}
    <path
      d="M 46 56 Q 50 60 54 56"
      fill="none"
      stroke={INK}
      strokeWidth="1.5"
      strokeLinecap="round"
    />

    {/* Cheek blush — very subtle */}
    <circle cx="29" cy="49" r="3.2" fill="#ffb6c1" opacity="0.45" />
    <circle cx="71" cy="49" r="3.2" fill="#ffb6c1" opacity="0.45" />
  </svg>
);

const PandaCompanion = () => {
  const [enabled, setEnabled] = useState(false);
  const [hasMoved, setHasMoved] = useState(false);
  const [isAsleep, setIsAsleep] = useState(false);
  const [blinking, setBlinking] = useState(false);

  // Random blinks every 3-6s for life. Each blink lasts ~120ms.
  useEffect(() => {
    if (!enabled) return undefined;
    let timeout;
    const scheduleBlink = () => {
      const wait = 3000 + Math.random() * 3000;
      timeout = setTimeout(() => {
        setBlinking(true);
        setTimeout(() => setBlinking(false), 120);
        scheduleBlink();
      }, wait);
    };
    scheduleBlink();
    return () => clearTimeout(timeout);
  }, [enabled]);

  // Media-query gate: only show on devices that have a real pointer and
  // haven't asked for reduced motion.
  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const hoverMQ = window.matchMedia('(hover: hover) and (pointer: fine)');
    const motionMQ = window.matchMedia('(prefers-reduced-motion: reduce)');
    const recompute = () => setEnabled(hoverMQ.matches && !motionMQ.matches);
    recompute();
    hoverMQ.addEventListener?.('change', recompute);
    motionMQ.addEventListener?.('change', recompute);
    return () => {
      hoverMQ.removeEventListener?.('change', recompute);
      motionMQ.removeEventListener?.('change', recompute);
    };
  }, []);

  // Spring-driven position — lags behind the cursor like a curious pet,
  // not glued to it.
  const targetX = useMotionValue(-100);
  const targetY = useMotionValue(-100);
  const springX = useSpring(targetX, { stiffness: 80, damping: 14, mass: 0.6 });
  const springY = useSpring(targetY, { stiffness: 80, damping: 14, mass: 0.6 });

  const idleTimer = useRef(null);

  useEffect(() => {
    if (!enabled) return undefined;

    const goSit = () => {
      const rx = window.innerWidth - SIZE - REST_PAD;
      const ry = window.innerHeight - SIZE - REST_PAD;
      targetX.set(rx);
      targetY.set(ry);
      setIsAsleep(true);
    };

    const onMove = (e) => {
      targetX.set(e.clientX + CURSOR_OFFSET);
      targetY.set(e.clientY + CURSOR_OFFSET);
      if (!hasMoved) setHasMoved(true);
      if (isAsleep) setIsAsleep(false);
      if (idleTimer.current) clearTimeout(idleTimer.current);
      idleTimer.current = setTimeout(goSit, IDLE_MS);
    };

    window.addEventListener('pointermove', onMove);
    return () => {
      window.removeEventListener('pointermove', onMove);
      if (idleTimer.current) clearTimeout(idleTimer.current);
    };
  }, [enabled, hasMoved, isAsleep, targetX, targetY]);

  if (!enabled) return null;

  return (
    <motion.div
      aria-hidden
      initial={{ opacity: 0 }}
      animate={{
        opacity: hasMoved ? 1 : 0,
        scale: isAsleep ? 0.92 : 1,
        rotate: isAsleep ? -6 : 0,
      }}
      transition={{
        opacity: { duration: 0.5, ease: 'easeOut' },
        scale: { duration: 0.55, ease: 'easeOut' },
        rotate: { duration: 0.55, ease: 'easeOut' },
      }}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: SIZE,
        height: SIZE,
        x: springX,
        y: springY,
        pointerEvents: 'none',
        zIndex: 9999,
        filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.28)) drop-shadow(0 2px 3px rgba(0,0,0,0.15))',
      }}
    >
      {/* Inner wrapper runs the breathing loop while sitting — separate
          from the outer transform (follow + tilt) so they don't compose. */}
      <motion.div
        style={{ width: '100%', height: '100%' }}
        animate={
          isAsleep
            ? { scale: [0.96, 1.04, 0.96] }
            : { scale: 1 }
        }
        transition={
          isAsleep
            ? { duration: 3.2, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.3 }
        }
      >
        <PandaSvg blinking={blinking} />
      </motion.div>
    </motion.div>
  );
};

export default PandaCompanion;
