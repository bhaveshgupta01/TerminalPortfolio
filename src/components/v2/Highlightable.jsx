import React, { useMemo } from 'react';
import { useV2Stack } from '../../context/V2StackContext';

/**
 * Wrap plain-text children. If any of the currently-highlighted phrases
 * appear inside, those matches get a `<mark>` that pulses — keyed off
 * `highlightPulseKey` so repeat AI commands re-trigger the animation.
 *
 * The mark's colour is intentionally white-on-dark so it works against any
 * card material's surface. Card-accent-tinted highlights would clash with
 * each other when phrases span sections of different materials.
 */

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const buildRegex = (phrases) => {
  const cleaned = phrases.map((p) => String(p || '').trim()).filter((p) => p.length > 1);
  if (!cleaned.length) return null;
  cleaned.sort((a, b) => b.length - a.length);
  return new RegExp(`(${cleaned.map(escapeRegex).join('|')})`, 'gi');
};

const Highlightable = ({ children }) => {
  const { highlightedPhrases, highlightPulseKey } = useV2Stack();
  const regex = useMemo(() => buildRegex(highlightedPhrases), [highlightedPhrases]);

  if (typeof children !== 'string' || !regex) {
    return <>{children}</>;
  }

  const parts = children.split(regex);
  return (
    <>
      {parts.map((part, i) => {
        if (!part) return null;
        const isMatch = i % 2 === 1;
        if (!isMatch) return <React.Fragment key={i}>{part}</React.Fragment>;
        return (
          <mark key={`${highlightPulseKey}-${i}`} className="v2-phrase-pulse">
            {part}
          </mark>
        );
      })}
    </>
  );
};

export default Highlightable;
