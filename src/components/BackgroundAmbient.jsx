import React from 'react';
import { motion } from 'framer-motion';

/**
 * Animated ambient background — 3 large, heavily-blurred radial orbs
 * that slowly drift on independent timelines. Sits behind all page content
 * (`z-0` + fixed positioning) so it provides constant, calm motion as the
 * user scrolls.
 *
 * Uses theme-aware accent tint at low alpha → works in both light & dark.
 */
const Orb = ({ className, color, duration, path }) => (
  <motion.div
    aria-hidden
    className={`absolute rounded-full pointer-events-none mix-blend-plus-lighter ${className}`}
    style={{
      background: `radial-gradient(circle, ${color} 0%, transparent 60%)`,
      willChange: 'transform',
    }}
    animate={path}
    transition={{ duration, repeat: Infinity, ease: 'easeInOut' }}
  />
);

const BackgroundAmbient = () => {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      {/* rose — top-left */}
      <Orb
        className="top-[-15%] left-[-10%] w-[65vw] h-[65vw] max-w-[900px] max-h-[900px]"
        color="rgba(251, 113, 133, 0.20)"
        duration={55}
        path={{
          x:     ['0%',  '12%',  '-4%',  '0%'],
          y:     ['0%',  '10%',  '18%',  '0%'],
          scale: [1,     1.1,    0.95,   1],
        }}
      />
      {/* amber — mid-right */}
      <Orb
        className="top-[25%] right-[-15%] w-[60vw] h-[60vw] max-w-[800px] max-h-[800px]"
        color="rgba(251, 191, 36, 0.17)"
        duration={72}
        path={{
          x:     ['0%',  '-18%', '8%',   '0%'],
          y:     ['0%',  '14%',  '-10%', '0%'],
          scale: [1,     0.92,   1.08,   1],
        }}
      />
      {/* blue — bottom-center */}
      <Orb
        className="bottom-[-20%] left-[15%] w-[55vw] h-[55vw] max-w-[750px] max-h-[750px]"
        color="rgba(59, 130, 246, 0.14)"
        duration={85}
        path={{
          x:     ['0%',  '18%',  '-8%',  '0%'],
          y:     ['0%',  '-12%', '6%',   '0%'],
          scale: [1,     1.06,   0.94,   1],
        }}
      />
    </div>
  );
};

export default BackgroundAmbient;
