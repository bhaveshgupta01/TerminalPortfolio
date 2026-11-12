import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

/**
 * v2 stack state.
 *
 * The URL is the source of truth for activeCard / openMode. setFromRoute()
 * mirrors a parsed URL into local state. Clicks and AI commands call
 * navigate() which delegates to whatever onNavigate the host provides
 * (V2Home pushes to React Router).
 *
 * Adds over the prior v2:
 *   - lastAgentMsg + dismissReply: drives the chat's reply-strip mode
 *     (option H-1 from spec — after a final lands, chat collapses to a
 *     one-line reply that the user clicks to dismiss).
 */

const V2StackContext = createContext(null);

const HIGHLIGHT_MS = 3200;

export const V2StackProvider = ({ children, onNavigate }) => {
  const [activeDeckId, setActiveDeckId] = useState('home');
  const [activeCardId, setActiveCardId] = useState(null);
  const [openMode, setOpenMode] = useState(false);
  const [activeSource, setActiveSource] = useState(null); // user | ai | route | palette
  const [focusedSection, setFocusedSection] = useState(null);
  const [highlightedPhrases, setHighlightedPhrases] = useState([]);
  const [highlightPulseKey, setHighlightPulseKey] = useState(0);

  // Agent / streaming
  const [agentStage, setAgentStage] = useState('idle');
  const [toolCalls, setToolCalls] = useState([]);
  const [lastAgentMsg, setLastAgentMsg] = useState(null); // string shown in reply strip
  // Buffering state — when ui_command lands but text hasn't started streaming
  // yet, show a shimmer overlay on the focused card so it doesn't look frozen.
  const [bufferingCardId, setBufferingCardId] = useState(null);
  // Focus reveal trigger — bumps each time AI focuses a card so we can
  // re-fire the sheen-sweep animation.
  const [focusRevealKey, setFocusRevealKey] = useState(0);

  const highlightTimer = useRef(null);

  useEffect(() => () => {
    if (highlightTimer.current) clearTimeout(highlightTimer.current);
  }, []);

  const setFromRoute = useCallback(({ deckId, cardId, open }) => {
    setActiveDeckId(deckId || 'home');
    setActiveCardId(cardId || null);
    setOpenMode(Boolean(open));
    setActiveSource('route');
  }, []);

  const navigate = useCallback((target, meta = {}) => {
    const next = {
      deckId: target.deckId ?? activeDeckId,
      cardId: target.cardId ?? null,
      open: Boolean(target.open),
    };
    setActiveSource(meta.source || 'user');
    if (typeof meta.sectionId === 'string') setFocusedSection(meta.sectionId);
    else if (meta.sectionId === null) setFocusedSection(null);
    if (Array.isArray(meta.highlight)) {
      setHighlightedPhrases(meta.highlight);
      setHighlightPulseKey((k) => k + 1);
      if (highlightTimer.current) clearTimeout(highlightTimer.current);
      highlightTimer.current = setTimeout(() => setHighlightedPhrases([]), HIGHLIGHT_MS);
    }
    onNavigate?.(next);
  }, [activeDeckId, onNavigate]);

  const applyFocusCommand = useCallback((payload) => {
    if (!payload || typeof payload !== 'object') return false;
    const { cardId, sectionId, highlight } = payload;
    if (!cardId) return false;
    // User lock — if user just opened a different card explicitly, respect it.
    if (activeSource === 'user' && activeCardId && activeCardId !== cardId) return false;
    navigate(
      { deckId: 'home', cardId, open: true },
      {
        source: 'ai',
        sectionId: sectionId || null,
        highlight: Array.isArray(highlight) ? highlight : [],
      },
    );
    setBufferingCardId(cardId);
    setFocusRevealKey((k) => k + 1);
    return true;
  }, [activeCardId, activeSource, navigate]);

  const clearBuffering = useCallback(() => setBufferingCardId(null), []);

  const pushToolCall = useCallback((tool) => {
    setToolCalls((prev) => [...prev, { ...tool, id: `${prev.length}-${tool.name}-${Date.now()}` }]);
  }, []);

  const resetTurn = useCallback(() => {
    setToolCalls([]);
    setAgentStage('idle');
  }, []);

  const dismissReply = useCallback(() => {
    setLastAgentMsg(null);
  }, []);

  const value = useMemo(() => ({
    // card state
    activeDeckId,
    activeCardId,
    openMode,
    activeSource,
    focusedSection,
    highlightedPhrases,
    highlightPulseKey,
    setFromRoute,
    navigate,
    applyFocusCommand,
    // agent state
    agentStage,
    setAgentStage,
    toolCalls,
    pushToolCall,
    resetTurn,
    lastAgentMsg,
    setLastAgentMsg,
    dismissReply,
    bufferingCardId,
    clearBuffering,
    focusRevealKey,
  }), [
    activeDeckId, activeCardId, openMode, activeSource,
    focusedSection, highlightedPhrases, highlightPulseKey,
    setFromRoute, navigate, applyFocusCommand,
    agentStage, toolCalls, pushToolCall, resetTurn,
    lastAgentMsg, dismissReply,
    bufferingCardId, clearBuffering, focusRevealKey,
  ]);

  return <V2StackContext.Provider value={value}>{children}</V2StackContext.Provider>;
};

export const useV2Stack = () => {
  const ctx = useContext(V2StackContext);
  if (!ctx) throw new Error('useV2Stack must be used inside V2StackProvider');
  return ctx;
};
