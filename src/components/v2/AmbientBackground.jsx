import React, { useEffect } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import LogoIcon from '../LogoIcon';
import { useV2Stack } from '../../context/V2StackContext';
import { getDeck } from './decks';
import { getMaterial } from './materials';

/**
 * Ambient v2 background.
 *
 *   - Faded brand watermark (LogoIcon) lives behind everything, gently
 *     drifts with mouse position.
 *   - Two warm orbs follow mouse direction (mid-amplitude parallax) so
 *     the page never feels still.
 *   - Per-card tint overlay: a soft radial of the active card's accent
 *     colour at the bottom of the viewport — crossfades when the active
 *     card changes, so scrolling through the stack feels like the room
 *     itself is changing temperature.
 */

const AmbientBackground = () => {
  const { activeDeckId, activeCardId } = useV2Stack();
  const deck = getDeck(activeDeckId);
  const activeCard =
    deck.cards.find((c) => c.id === activeCardId) || deck.cards[0];
  const mat = getMaterial(activeCard?.material);
  const tint = mat.tint;
  const accent = mat.accent;

  // Mouse → spring-smoothed normalized motion values
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const xs = useSpring(mx, { damping: 26, stiffness: 80, mass: 0.6 });
  const ys = useSpring(my, { damping: 26, stiffness: 80, mass: 0.6 });

  useEffect(() => {
    const onMove = (e) => {
      const cx = window.innerWidth / 2;
      const cy = window.innerHeight / 2;
      mx.set((e.clientX - cx) / cx);
      my.set((e.clientY - cy) / cy);
    };
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, [mx, my]);

  // Different cards in the stack get different parallax amplitudes so the
  // motion feels layered, not unison.
  const orb1X = useTransform(xs, [-1, 1], [-32, 32]);
  const orb1Y = useTransform(ys, [-1, 1], [-22, 22]);
  const orb2X = useTransform(xs, [-1, 1], [44, -44]);
  const orb2Y = useTransform(ys, [-1, 1], [30, -30]);
  const orb3X = useTransform(xs, [-1, 1], [-18, 18]);
  const orb3Y = useTransform(ys, [-1, 1], [22, -22]);
  const logoX = useTransform(xs, [-1, 1], [-14, 14]);
  const logoY = useTransform(ys, [-1, 1], [-10, 10]);
  const logoRotate = useTransform(xs, [-1, 1], [-2, 2]);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    >
      {/* Static base wash — layered radials for depth. */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `
            radial-gradient(ellipse 100% 70% at 50% 0%, var(--v2-vignette-top), transparent 65%),
            radial-gradient(ellipse 90% 60% at 50% 100%, var(--v2-vignette-bot), transparent 70%)
          `,
          transition: 'background 600ms ease',
        }}
      />

      {/* Brand watermark — much larger, slowly breathes + drifts with cursor.
          The signature mark is the spine of the page: it should fill the
          canvas. Opacity 0.10 keeps it ambient, not assertive. */}
      <motion.div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '110vmin',
          height: '110vmin',
          marginLeft: '-55vmin',
          marginTop: '-55vmin',
          opacity: 'var(--v2-watermark-opacity)',
          x: logoX,
          y: logoY,
          color: tint,
        }}
        animate={{
          scale: [1, 1.025, 1],
          rotate: [-1.2, 1.2, -1.2],
        }}
        transition={{
          scale:  { duration: 14, repeat: Infinity, ease: 'easeInOut' },
          rotate: { duration: 22, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        <LogoIcon className="w-full h-full" strokeWidth={22} />
      </motion.div>

      {/* Inner brighter version of the watermark — gives it dimension */}
      <motion.div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '70vmin',
          height: '70vmin',
          marginLeft: '-35vmin',
          marginTop: '-35vmin',
          opacity: 0.05,
          x: logoX,
          y: logoY,
          rotate: logoRotate,
          color: accent,
          filter: 'blur(12px)',
          mixBlendMode: 'screen',
        }}
      >
        <LogoIcon className="w-full h-full" strokeWidth={36} />
      </motion.div>

      {/* Orb 1 — top-left, takes the active card's tint colour */}
      <motion.div
        style={{
          position: 'absolute',
          top: '-18%',
          left: '-12%',
          width: '60vw',
          height: '60vw',
          maxWidth: 900,
          maxHeight: 900,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${tint}3a 0%, transparent 60%)`,
          filter: 'blur(60px)',
          x: orb1X,
          y: orb1Y,
          willChange: 'transform',
        }}
      />

      {/* Orb 2 — bottom-right, accent colour */}
      <motion.div
        style={{
          position: 'absolute',
          bottom: '-18%',
          right: '-12%',
          width: '55vw',
          height: '55vw',
          maxWidth: 800,
          maxHeight: 800,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${accent}26 0%, transparent 60%)`,
          filter: 'blur(60px)',
          x: orb2X,
          y: orb2Y,
          willChange: 'transform',
        }}
      />

      {/* Orb 3 — small warm spotlight, mid-screen */}
      <motion.div
        style={{
          position: 'absolute',
          top: '40%',
          left: '20%',
          width: '30vw',
          height: '30vw',
          maxWidth: 500,
          maxHeight: 500,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${tint}1c 0%, transparent 60%)`,
          filter: 'blur(50px)',
          x: orb3X,
          y: orb3Y,
          willChange: 'transform',
        }}
      />

      {/* Per-card tint overlay — crossfades with activeCardId via key */}
      <motion.div
        key={activeCardId || 'default'}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: 'easeOut' }}
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(ellipse 90% 70% at 50% 110%, ${tint}33, transparent 72%)`,
          mixBlendMode: 'soft-light',
        }}
      />
    </div>
  );
};

export default AmbientBackground;
