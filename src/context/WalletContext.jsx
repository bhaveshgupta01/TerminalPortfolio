import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';

export const WALLET_CARD_IDS = [
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

export const isWalletCardId = (value) => WALLET_CARD_IDS.includes(value);

const WalletContext = createContext(null);

export const WalletProvider = ({ children, initialCard = null }) => {
  const safeInitialCard = isWalletCardId(initialCard) ? initialCard : null;
  const [activeCard, setActiveCardState] = useState(safeInitialCard);
  const [activeCardSource, setActiveCardSource] = useState(safeInitialCard ? 'route' : null);
  const [highlightedCard, setHighlightedCard] = useState(safeInitialCard);
  const highlightTimerRef = useRef(null);

  useEffect(() => () => {
    if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
  }, []);

  const flashCard = useCallback((cardId) => {
    if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
    setHighlightedCard(cardId);
    highlightTimerRef.current = setTimeout(() => setHighlightedCard(null), 2200);
  }, []);

  const setActiveCard = useCallback((cardId, options = {}) => {
    if (!isWalletCardId(cardId)) return;
    const source = options.source || 'user';
    const highlight = options.highlight ?? source === 'ai';

    setActiveCardState(cardId);
    setActiveCardSource(source);

    if (highlight) flashCard(cardId);
  }, [flashCard]);

  const clearActiveCard = useCallback(() => {
    setActiveCardState(null);
    setActiveCardSource(null);
  }, []);

  return (
    <WalletContext.Provider
      value={{
        activeCard,
        activeCardSource,
        highlightedCard,
        setActiveCard,
        clearActiveCard,
      }}
    >
      {children}
    </WalletContext.Provider>
  );
};

export const useWallet = () => {
  const value = useContext(WalletContext);
  if (!value) throw new Error('useWallet must be used inside WalletProvider');
  return value;
};
