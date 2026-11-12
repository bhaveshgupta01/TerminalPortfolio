import React, { useMemo } from 'react';
import { useLabWallet } from '../../context/LabWalletContext';

/**
 * Wraps a plain-text child and, if any of the currently-highlighted phrases
 * appear inside it (case-insensitive), wraps those matches in a <mark> that
 * pulses — driven by the `pulseKey` from the wallet context so repeat
 * commands re-trigger the animation.
 *
 * Opt-in: only text you explicitly render through <Highlightable> can be
 * targeted. Protects against the AI pulsing things like navigation chrome.
 */

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const buildRegex = (phrases) => {
  const cleaned = phrases
    .map((p) => String(p || '').trim())
    .filter((p) => p.length > 1);
  if (!cleaned.length) return null;
  // Sort by length descending so longer phrases win over shorter substrings.
  cleaned.sort((a, b) => b.length - a.length);
  return new RegExp(`(${cleaned.map(escapeRegex).join('|')})`, 'gi');
};

const Highlightable = ({ children, as: Tag = React.Fragment }) => {
  const { highlightedPhrases, highlightPulseKey } = useLabWallet();
  const regex = useMemo(() => buildRegex(highlightedPhrases), [highlightedPhrases]);

  if (typeof children !== 'string' || !regex) {
    return Tag === React.Fragment ? <>{children}</> : <Tag>{children}</Tag>;
  }

  const parts = children.split(regex);
  const nodes = parts.map((part, i) => {
    if (!part) return null;
    // Odd indices are matches because the regex uses a capturing group.
    const isMatch = i % 2 === 1;
    if (!isMatch) return <React.Fragment key={i}>{part}</React.Fragment>;
    return (
      <mark
        key={`${highlightPulseKey}-${i}`}
        className="lab-phrase-pulse"
      >
        {part}
      </mark>
    );
  });

  return Tag === React.Fragment ? <>{nodes}</> : <Tag>{nodes}</Tag>;
};

export default Highlightable;
