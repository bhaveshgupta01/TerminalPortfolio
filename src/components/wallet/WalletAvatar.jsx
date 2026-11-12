import React from 'react';
import { motion } from 'framer-motion';
import BrandMark from '../BrandMark';

const STATE_MAP = {
  idle: {
    glow: 'rgba(244, 63, 94, 0.18)',
    ring: 'rgba(244, 63, 94, 0.28)',
    label: 'Idle',
  },
  thinking: {
    glow: 'rgba(251, 191, 36, 0.26)',
    ring: 'rgba(251, 191, 36, 0.34)',
    label: 'Thinking',
  },
  guiding: {
    glow: 'rgba(59, 130, 246, 0.22)',
    ring: 'rgba(59, 130, 246, 0.30)',
    label: 'Guiding',
  },
  replied: {
    glow: 'rgba(16, 185, 129, 0.24)',
    ring: 'rgba(16, 185, 129, 0.32)',
    label: 'Focused',
  },
};

const WalletAvatar = ({ state = 'idle' }) => {
  const meta = STATE_MAP[state] || STATE_MAP.idle;

  return (
    <div className="relative flex flex-col items-center gap-3">
      <motion.div
        aria-hidden
        className="absolute inset-0 rounded-full blur-2xl"
        animate={{
          scale: state === 'thinking' ? [1, 1.15, 1] : [1, 1.05, 1],
          opacity: state === 'thinking' ? [0.6, 1, 0.6] : [0.55, 0.8, 0.55],
          background: [
            `radial-gradient(circle, ${meta.glow} 0%, transparent 70%)`,
            `radial-gradient(circle, ${meta.glow} 0%, transparent 70%)`,
          ],
        }}
        transition={{ duration: state === 'thinking' ? 1.1 : 2.2, repeat: Infinity }}
      />

      <motion.div
        className="relative flex h-20 w-20 items-center justify-center rounded-full border border-subtle/20 bg-surface/65"
        animate={{
          boxShadow: [
            `0 0 0 0 ${meta.ring}`,
            `0 0 0 18px rgba(0,0,0,0)`,
          ],
          scale: state === 'thinking' ? [1, 1.03, 1] : 1,
        }}
        transition={{ duration: state === 'thinking' ? 1.15 : 2.4, repeat: Infinity }}
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

export default WalletAvatar;
