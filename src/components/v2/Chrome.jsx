import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useV2Stack } from '../../context/V2StackContext';
import { useTheme } from '../../context/ThemeContext';
import { getDeck } from './decks';

/**
 * Persistent screen chrome — pills, hint, and progress dots that frame the
 * stack zone. All hide when a card is focused so the detail view feels
 * uncluttered. Surface colours come from v2 CSS variables so they flip with
 * the light/dark theme; the metallic card faces (Card.jsx) stay unchanged.
 */

export const IdentityPill = ({ isMobile }) => (
  <div
    style={{
      position: 'absolute',
      bottom: `max(${isMobile ? 16 : 22}px, env(safe-area-inset-bottom))`,
      left: `max(${isMobile ? 14 : 24}px, env(safe-area-inset-left))`,
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '7px 14px 7px 7px',
      borderRadius: 999,
      background: 'var(--v2-pill-bg)',
      backdropFilter: 'blur(20px)',
      border: '1px solid var(--v2-pill-border)',
      zIndex: 80,
      pointerEvents: 'none',
      transition: 'background 400ms ease, border-color 400ms ease',
    }}
  >
    <div
      style={{
        width: 24,
        height: 24,
        borderRadius: '50%',
        overflow: 'hidden',
        border: '1px solid var(--v2-pill-border-hi)',
      }}
    >
      <img src="/panda.png" alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
    </div>
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: 'var(--v2-ink)',
          fontFamily: "'Inter Tight', sans-serif",
        }}
      >
        {isMobile ? 'Bhavesh' : 'Bhavesh Gupta'}
      </div>
      {!isMobile && (
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            letterSpacing: 1,
            color: 'var(--v2-ink-mute)',
            textTransform: 'uppercase',
          }}
        >
          AI · ENG
        </div>
      )}
    </div>
  </div>
);

export const AvailabilityPill = ({ isMobile }) => (
  <div
    style={{
      position: 'absolute',
      bottom: `max(${isMobile ? 16 : 22}px, env(safe-area-inset-bottom))`,
      right: `max(${isMobile ? 14 : 48}px, env(safe-area-inset-right))`,
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '6px 12px',
      borderRadius: 999,
      background: 'var(--v2-avail-bg)',
      backdropFilter: 'blur(20px)',
      border: '1px solid var(--v2-avail-border)',
      zIndex: 80,
      transition: 'background 400ms ease, border-color 400ms ease',
    }}
  >
    <div
      style={{
        width: 7,
        height: 7,
        borderRadius: '50%',
        background: 'var(--v2-avail-dot)',
        boxShadow: '0 0 8px var(--v2-avail-dot)',
      }}
    />
    <div
      style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 10,
        letterSpacing: 1,
        color: 'var(--v2-avail-text)',
        textTransform: 'uppercase',
      }}
    >
      {isMobile ? 'Open' : 'Open to summer 2026'}
    </div>
  </div>
);

export const ScrollHint = ({ isMobile }) => (
  <div
    style={{
      position: 'absolute',
      top: 24,
      left: isMobile ? '50%' : 24,
      transform: isMobile ? 'translateX(-50%)' : 'none',
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '6px 12px',
      borderRadius: 999,
      background: 'var(--v2-pill-bg)',
      backdropFilter: 'blur(20px)',
      border: '1px solid var(--v2-pill-border)',
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 10,
      letterSpacing: 1.5,
      color: 'var(--v2-ink-faint)',
      textTransform: 'uppercase',
      zIndex: 50,
      pointerEvents: 'none',
      animation: 'fadeIn 0.8s ease',
      whiteSpace: 'nowrap',
      transition: 'background 400ms ease, border-color 400ms ease, color 400ms ease',
    }}
  >
    <span style={{ display: 'inline-block', animation: 'v2-hint-bounce 1.8s ease-in-out infinite' }}>↓</span>
    <span>{isMobile ? 'swipe · tap top card' : 'scroll to browse · click to open'}</span>
  </div>
);

export const ProgressDots = () => {
  const { activeDeckId, activeCardId, navigate } = useV2Stack();
  const deck = getDeck(activeDeckId);
  const cards = deck.cards;
  const activeIndex = Math.max(0, cards.findIndex((c) => c.id === activeCardId));

  return (
    <div
      style={{
        position: 'absolute',
        right: 24,
        top: '50%',
        transform: 'translateY(-50%)',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        zIndex: 50,
      }}
    >
      {cards.map((c, i) => (
        <button
          key={c.id}
          type="button"
          onClick={() => navigate({ deckId: activeDeckId, cardId: c.id, open: false }, { source: 'user' })}
          aria-label={`jump to ${c.title}`}
          title={c.label}
          style={{
            width: 4,
            height: i === activeIndex ? 22 : 4,
            borderRadius: 2,
            background:
              i === activeIndex
                ? 'var(--v2-dot-active)'
                : 'var(--v2-dot-idle)',
            border: 'none',
            padding: 0,
            cursor: 'pointer',
            transition: 'all 0.35s cubic-bezier(0.32, 0.72, 0, 1), background 400ms ease',
          }}
        />
      ))}
    </div>
  );
};

const chromeButtonStyle = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 40,
  height: 40,
  borderRadius: 12,
  background: 'var(--v2-pill-bg)',
  backdropFilter: 'blur(20px)',
  border: '1px solid var(--v2-pill-border)',
  cursor: 'pointer',
  transition: 'background 200ms ease, border-color 200ms ease, color 400ms ease',
  color: 'var(--v2-ink)',
  textDecoration: 'none',
};

const hoverIn = (e) => {
  e.currentTarget.style.background = 'var(--v2-hover-bg)';
  e.currentTarget.style.border = '1px solid var(--v2-pill-border-hi)';
};
const hoverOut = (e) => {
  e.currentTarget.style.background = 'var(--v2-pill-bg)';
  e.currentTarget.style.border = '1px solid var(--v2-pill-border)';
};

/**
 * Top-right cluster: theme toggle + home button.
 * - z-index 200 sits above the chat bar (z-index 100)
 * - safe-area-inset-* keeps the cluster off notches/Dynamic Island
 * - flex container so both buttons shift together on landscape iPhone
 */
export const TopRightChrome = ({ isMobile }) => {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';
  const baseTop = isMobile ? 16 : 24;
  const baseRight = isMobile ? 14 : 24;

  return (
    <div
      style={{
        position: 'absolute',
        top: `max(${baseTop}px, env(safe-area-inset-top))`,
        right: `max(${baseRight}px, env(safe-area-inset-right))`,
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}
    >
      <button
        type="button"
        onClick={toggle}
        aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
        title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
        style={chromeButtonStyle}
        onMouseEnter={hoverIn}
        onMouseLeave={hoverOut}
      >
        {isDark ? <Moon size={16} /> : <Sun size={16} />}
      </button>
      <a
        href="/"
        title="Go to home"
        style={{ ...chromeButtonStyle, fontSize: 20 }}
        onMouseEnter={hoverIn}
        onMouseLeave={hoverOut}
      >
        ⌂
      </a>
    </div>
  );
};

// Back-compat re-exports — pages still importing the old names won't break.
export const HomeButton = ({ isMobile }) => <TopRightChrome isMobile={isMobile} />;
export const ThemeToggleButton = () => null;
