import React from 'react';
import LogoIcon from './LogoIcon';

/**
 * Brand mark — uses the geometric LogoIcon. Auto-adapts to theme via
 * `currentColor` + the `text-ink` class. A soft rose→amber glow sits behind
 * to tie the mark into the portfolio's accent palette.
 */
const BrandMark = ({ size = 36, className = '', glow = true, strokeWidth = 55 }) => (
  <div
    className={`relative flex items-center justify-center rounded-xl shrink-0 ${className}`}
    style={{ width: size, height: size }}
  >
    {glow && (
      <div
        aria-hidden
        className="absolute inset-0 rounded-full bg-gradient-to-br from-rose-400 to-amber-300 opacity-40 blur-md"
      />
    )}
    <LogoIcon
      className="relative z-10 w-[82%] h-[82%] text-ink"
      strokeWidth={strokeWidth}
    />
  </div>
);

export default BrandMark;
