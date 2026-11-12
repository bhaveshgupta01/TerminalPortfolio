import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';

/**
 * Panda avatar — uses /panda.png with no ring around it.
 *
 * Always animates subtly:
 *   idle               — gentle breathing scale + slow head bob
 *   thinking           — faster breathe + slight wobble + amber accent dot
 *   searching_profile  — slight tilt cycle + blue accent dot
 *   searching_web      — slight tilt the other way + emerald dot
 *   speaking           — small head bob + rose dot pulse
 *   done               — settles, emerald dot
 *   error              — still, red dot pulse
 *
 * The accent comes from a tiny dot in the bottom-right corner (live status
 * indicator) — no ring around the panda itself.
 */

const STATE = {
  idle:              { dot: '#7cd4a0', dotPulse: false, breathe: 3.6, tilt: 0,  bob: 0,    glowOpacity: 0 },
  thinking:          { dot: '#f4b860', dotPulse: true,  breathe: 1.4, tilt: 2,  bob: 1.5,  glowOpacity: 0.35 },
  searching_profile: { dot: '#90b0e4', dotPulse: true,  breathe: 1.6, tilt: 4,  bob: 0,    glowOpacity: 0.30 },
  searching_web:     { dot: '#7cd4a0', dotPulse: true,  breathe: 1.6, tilt: -4, bob: 0,    glowOpacity: 0.30 },
  tool_done:         { dot: '#7cd4a0', dotPulse: false, breathe: 2.4, tilt: 0,  bob: 0,    glowOpacity: 0.18 },
  speaking:          { dot: '#e490a8', dotPulse: true,  breathe: 1.1, tilt: 0,  bob: 2.5,  glowOpacity: 0.40 },
  done:              { dot: '#7cd4a0', dotPulse: false, breathe: 3.6, tilt: 0,  bob: 0,    glowOpacity: 0.20 },
  error:             { dot: '#e07070', dotPulse: true,  breathe: 3.6, tilt: 0,  bob: 0,    glowOpacity: 0.30 },
};

const PandaAvatar = ({ stage = 'idle', size = 36, showStatus = true }) => {
  const meta = STATE[stage] || STATE.idle;
  const reduced = useReducedMotion();

  const breatheTransition = reduced
    ? { duration: 0 }
    : { duration: meta.breathe, repeat: Infinity, ease: 'easeInOut' };

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        flexShrink: 0,
      }}
    >
      {/* very soft accent halo — colour-matched to the dot */}
      {!reduced && meta.glowOpacity > 0 && (
        <motion.div
          aria-hidden
          style={{
            position: 'absolute',
            inset: -size * 0.25,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${meta.dot}55 0%, transparent 65%)`,
            pointerEvents: 'none',
          }}
          animate={{ opacity: [meta.glowOpacity * 0.6, meta.glowOpacity, meta.glowOpacity * 0.6] }}
          transition={breatheTransition}
        />
      )}

      <motion.img
        src="/panda.png"
        alt="Bhavesh.ai"
        draggable={false}
        animate={{
          scale: reduced ? 1 : [1, 1.04, 1],
          rotate: reduced ? 0 : [0, meta.tilt, -meta.tilt, 0],
          y: reduced ? 0 : [0, -meta.bob, 0],
        }}
        transition={breatheTransition}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          borderRadius: '50%',
          display: 'block',
        }}
      />

      {showStatus && (
        <motion.span
          aria-hidden
          animate={meta.dotPulse && !reduced
            ? { scale: [1, 1.25, 1], opacity: [1, 0.7, 1] }
            : { scale: 1, opacity: 1 }
          }
          transition={meta.dotPulse && !reduced
            ? { duration: 1.1, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0 }
          }
          style={{
            position: 'absolute',
            bottom: -1,
            right: -1,
            width: Math.max(8, size * 0.22),
            height: Math.max(8, size * 0.22),
            borderRadius: '50%',
            background: meta.dot,
            border: '2px solid var(--v2-bg-deep)',
            boxShadow: `0 0 6px ${meta.dot}aa`,
          }}
        />
      )}
    </div>
  );
};

export default PandaAvatar;
