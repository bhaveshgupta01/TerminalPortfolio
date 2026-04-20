import React, { useEffect } from 'react';
import { motion, useMotionValue, useTransform, useSpring } from 'framer-motion';
import LogoIcon from './LogoIcon';

/**
 * Scroll-driven brand watermark — inspired by scroll-driven-animations.style/.
 *
 * Tracks the internal scroll container's scrollTop directly. Maps the first
 * viewport of scroll (hero exit) onto scale/rotate/translate/opacity, so the
 * clean geometric logo grows from a small mark into a large, faded, tilted
 * watermark behind the page content.
 */
const ScrollDrivenLogo = ({ scrollContainerRef }) => {
  const progress = useMotionValue(0);
  const smooth = useSpring(progress, { stiffness: 120, damping: 24, mass: 0.6 });

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const update = () => {
      const heroHeight = el.clientHeight || window.innerHeight;
      const p = Math.min(Math.max(el.scrollTop / heroHeight, 0), 1);
      progress.set(p);
    };
    update();
    el.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    return () => {
      el.removeEventListener('scroll', update);
      window.removeEventListener('resize', update);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scale   = useTransform(smooth, [0, 1],            [0.8, 3.6]);
  const rotate  = useTransform(smooth, [0, 1],            [0,    -6]);
  const ty      = useTransform(smooth, [0, 1],            ['-1vh', '3vh']);
  const opacity = useTransform(smooth, [0, 0.1, 0.9, 1],  [0.22, 0.18, 0.08, 0.06]);

  return (
    <div
      aria-hidden
      className="fixed inset-0 pointer-events-none flex items-center justify-center"
      style={{ zIndex: 0 }}
    >
      <motion.div
        style={{
          scale,
          rotate,
          y: ty,
          opacity,
          width:  'min(42vw, 560px)',
          height: 'min(42vw, 560px)',
          willChange: 'transform, opacity',
        }}
      >
        <LogoIcon
          className="w-full h-full text-ink"
          strokeWidth={36}
        />
      </motion.div>
    </div>
  );
};

export default ScrollDrivenLogo;
