import React from 'react';
import { motion } from 'framer-motion';
import BrandMark from '../BrandMark';

const STATE_MAP = {
  idle: {
    glow: 'rgba(244, 63, 94, 0.18)',
    ring: 'rgba(244, 63, 94, 0.28)',
    label: 'Idle',
    pulse: 2.4,
  },
  thinking: {
    glow: 'rgba(251, 191, 36, 0.26)',
    ring: 'rgba(251, 191, 36, 0.34)',
    label: 'Thinking',
    pulse: 1.05,
  },
  searching_profile: {
    glow: 'rgba(59, 130, 246, 0.28)',
    ring: 'rgba(59, 130, 246, 0.32)',
    label: 'Reading my notes',
    pulse: 1.2,
  },
  searching_web: {
    glow: 'rgba(16, 185, 129, 0.28)',
    ring: 'rgba(16, 185, 129, 0.34)',
    label: 'Searching the web',
    pulse: 1.2,
  },
  speaking: {
    glow: 'rgba(244, 63, 94, 0.30)',
    ring: 'rgba(244, 63, 94, 0.36)',
    label: 'Speaking',
    pulse: 0.9,
  },
  done: {
    glow: 'rgba(16, 185, 129, 0.24)',
    ring: 'rgba(16, 185, 129, 0.32)',
    label: 'Focused',
    pulse: 2.4,
  },
  error: {
    glow: 'rgba(239, 68, 68, 0.30)',
    ring: 'rgba(239, 68, 68, 0.36)',
    label: 'Offline',
    pulse: 2.4,
  },
};

const LabAvatar = ({ stage = 'idle' }) => {
  const meta = STATE_MAP[stage] || STATE_MAP.idle;
  const isActive = stage === 'thinking' || stage === 'searching_profile' || stage === 'searching_web' || stage === 'speaking';

  return (
    <div className="relative flex flex-col items-center gap-3">
      <motion.div
        aria-hidden
        className="absolute inset-0 rounded-full blur-2xl"
        animate={{
          scale: isActive ? [1, 1.12, 1] : [1, 1.04, 1],
          opacity: isActive ? [0.6, 1, 0.6] : [0.55, 0.8, 0.55],
          background: [
            `radial-gradient(circle, ${meta.glow} 0%, transparent 70%)`,
            `radial-gradient(circle, ${meta.glow} 0%, transparent 70%)`,
          ],
        }}
        transition={{ duration: meta.pulse, repeat: Infinity }}
      />

      <motion.div
        className="relative flex h-20 w-20 items-center justify-center rounded-full border border-subtle/20 bg-surface/65"
        animate={{
          boxShadow: [
            `0 0 0 0 ${meta.ring}`,
            `0 0 0 18px rgba(0,0,0,0)`,
          ],
          scale: isActive ? [1, 1.03, 1] : 1,
        }}
        transition={{ duration: meta.pulse, repeat: Infinity }}
        aria-label={`agent ${meta.label}`}
      >
        <BrandMark size={46} />
      </motion.div>

      <div className="text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-faint">
          Bhavesh.ai
        </div>
        <div className="mt-1 text-xs text-muted">{meta.label}</div>
      </div>
    </div>
  );
};

export default LabAvatar;
