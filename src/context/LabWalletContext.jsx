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
 * Lab wallet state.
 *
 * Richer than the phase-1 WalletContext:
 *   - activeCard            which card is open
 *   - activeCardSource      who opened it: "user" | "ai" | "route" | "palette"
 *   - focusedSection        optional sub-section id inside the card
 *   - highlightedPhrases    phrases to pulse-mark inside that section
 *   - agentStage            current streaming stage (idle, thinking,
 *                           searching_profile, searching_web, speaking, done,
 *                           error)
 *   - toolCalls             log of tool calls emitted during the current turn
 *
 * Callers:
 *   - user click, palette select, URL route, keyboard nav  → setActiveCard(id, { source })
 *   - AI SET_ACTIVE_CARD command                          → applyFocusCommand(payload)
 *   - stream listener                                      → setAgentStage, pushToolCall, …
 */

export const LAB_CARD_IDS = [
  'about',
  'journey',
  'experience',
  'skills',
  'projects',
  'leadership',
  'publications',
  'design',
  'contact',
];

export const isLabCardId = (value) => LAB_CARD_IDS.includes(value);

const FOCUS_HIGHLIGHT_MS = 2800;

const LabWalletContext = createContext(null);

export const LabWalletProvider = ({
  children,
  initialCard = null,
  initialSection = null,
}) => {
  const safeInitialCard = isLabCardId(initialCard) ? initialCard : null;
  const [activeCard, setActiveCardState] = useState(safeInitialCard);
  const [activeCardSource, setActiveCardSource] = useState(
    safeInitialCard ? 'route' : null,
  );
  const [focusedSection, setFocusedSection] = useState(
    safeInitialCard ? initialSection || null : null,
  );
  const [highlightedPhrases, setHighlightedPhrases] = useState([]);
  const [highlightPulseKey, setHighlightPulseKey] = useState(0);
  const [agentStage, setAgentStage] = useState('idle');
  const [toolCalls, setToolCalls] = useState([]);
  const [lastCommandAt, setLastCommandAt] = useState(0);

  const clearHighlightTimer = useRef(null);

  useEffect(() => {
    return () => {
      if (clearHighlightTimer.current) clearTimeout(clearHighlightTimer.current);
    };
  }, []);

  const setActiveCard = useCallback((cardId, options = {}) => {
    if (!isLabCardId(cardId)) return;
    const source = options.source || 'user';
    setActiveCardState(cardId);
    setActiveCardSource(source);
    setFocusedSection(options.sectionId || null);
    setHighlightedPhrases(Array.isArray(options.highlight) ? options.highlight : []);
    if (options.highlight && options.highlight.length) {
      setHighlightPulseKey((k) => k + 1);
    }
    setLastCommandAt(Date.now());
  }, []);

  const clearActiveCard = useCallback(() => {
    setActiveCardState(null);
    setActiveCardSource(null);
    setFocusedSection(null);
    setHighlightedPhrases([]);
  }, []);

  /**
   * Apply a ui_command.SET_ACTIVE_CARD payload from the agent, unless the
   * user has manually locked a different card in this turn (in which case
   * we respect the user's choice).
   */
  const applyFocusCommand = useCallback((payload, meta = {}) => {
    if (!payload || !isLabCardId(payload.cardId)) return false;
    const { respectUserLock = true } = meta;
    if (
      respectUserLock &&
      activeCardSource === 'user' &&
      activeCard &&
      activeCard !== payload.cardId
    ) {
      return false;
    }

    setActiveCardState(payload.cardId);
    setActiveCardSource('ai');
    setFocusedSection(payload.sectionId || null);
    setHighlightedPhrases(Array.isArray(payload.highlight) ? payload.highlight : []);
    setHighlightPulseKey((k) => k + 1);
    setLastCommandAt(Date.now());

    if (clearHighlightTimer.current) clearTimeout(clearHighlightTimer.current);
    clearHighlightTimer.current = setTimeout(() => {
      setHighlightedPhrases([]);
    }, FOCUS_HIGHLIGHT_MS);

    return true;
  }, [activeCard, activeCardSource]);

  const pushToolCall = useCallback((tool) => {
    setToolCalls((prev) => [...prev, { ...tool, id: `${prev.length}-${tool.name}-${Date.now()}` }]);
  }, []);

  const resetTurn = useCallback(() => {
    setToolCalls([]);
    setAgentStage('idle');
  }, []);

  const value = useMemo(() => ({
    // card state
    activeCard,
    activeCardSource,
    focusedSection,
    highlightedPhrases,
    highlightPulseKey,
    lastCommandAt,
    setActiveCard,
    clearActiveCard,
    applyFocusCommand,
    // agent stream state
    agentStage,
    setAgentStage,
    toolCalls,
    pushToolCall,
    resetTurn,
  }), [
    activeCard,
    activeCardSource,
    focusedSection,
    highlightedPhrases,
    highlightPulseKey,
    lastCommandAt,
    setActiveCard,
    clearActiveCard,
    applyFocusCommand,
    agentStage,
    toolCalls,
    pushToolCall,
    resetTurn,
  ]);

  return (
    <LabWalletContext.Provider value={value}>
      {children}
    </LabWalletContext.Provider>
  );
};

export const useLabWallet = () => {
  const ctx = useContext(LabWalletContext);
  if (!ctx) throw new Error('useLabWallet must be used inside LabWalletProvider');
  return ctx;
};
