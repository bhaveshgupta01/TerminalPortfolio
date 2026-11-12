import React, {
  startTransition,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  User,
  Briefcase,
  Cpu,
  Folder,
  Palette,
  Mail,
  BookOpen,
  X,
} from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import BackgroundAmbient from '../components/BackgroundAmbient';
import ThemeToggle from '../components/ThemeToggle';
import WalletChatPanel from '../components/wallet/WalletChatPanel';
import WalletStack from '../components/wallet/WalletStack';
import { WALLET_CARD_DEFS } from '../components/wallet/walletContent';
import {
  WalletProvider,
  isWalletCardId,
  useWallet,
} from '../context/WalletContext';

const API_URL = import.meta.env.VITE_AGENT_API || 'http://localhost:8000';

const CARD_MAP = Object.fromEntries(WALLET_CARD_DEFS.map((card) => [card.id, card]));

const suggestionChips = [
  { label: 'About Me', cmd: 'Who are you?', icon: <User size={14} /> },
  { label: 'Experience', cmd: 'Walk me through your work experience.', icon: <Briefcase size={14} /> },
  { label: 'Projects', cmd: 'What are your favorite projects?', icon: <Folder size={14} /> },
  { label: 'Tech Stack', cmd: 'What is your tech stack?', icon: <Cpu size={14} /> },
  { label: 'Design', cmd: 'design', icon: <Palette size={14} /> },
  { label: 'Publications', cmd: 'Tell me about your published research.', icon: <BookOpen size={14} /> },
  { label: 'Contact', cmd: 'How do I contact you?', icon: <Mail size={14} /> },
  { label: 'Clear', cmd: 'clear', icon: <X size={14} /> },
];

const WalletHomeScreen = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    activeCard,
    activeCardSource,
    highlightedCard,
    setActiveCard,
    clearActiveCard,
  } = useWallet();

  const [isTyping, setIsTyping] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const chatBottomRef = useRef(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    const nextParams = new URLSearchParams(searchParams);
    if (activeCard) {
      nextParams.set('card', activeCard);
    } else {
      nextParams.delete('card');
    }

    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true });
    }
  }, [activeCard, searchParams, setSearchParams]);

  const handleCardSelect = (cardId) => {
    setActiveCard(cardId, { source: 'user' });
  };

  const handleBackendFocus = (cardId) => {
    if (!isWalletCardId(cardId)) return;
    if (activeCardSource === 'user' && activeCard) return;

    startTransition(() => {
      setActiveCard(cardId, { source: 'ai', highlight: true });
    });
  };

  const handleSend = async (text = input) => {
    const q = (text || '').trim();
    if (!q) return;

    if (q.toLowerCase() === 'clear') {
      setMessages([]);
      setInput('');
      clearActiveCard();
      return;
    }

    if (q.toLowerCase() === 'design') {
      setInput('');
      setActiveCard('design', { source: 'user' });
      return;
    }

    const userMsg = { id: Date.now(), role: 'user', text: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, thread_id: threadId }),
      });
      if (!res.ok) throw new Error(`${res.status}`);

      const data = await res.json();

      if (data.thread_id) setThreadId(data.thread_id);
      if (data.scroll_to) handleBackendFocus(data.scroll_to);

      setMessages((prev) => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: data.answer || '…',
        component: data.component || null,
        source: data.source || 'profile',
      }]);
    } catch (error) {
      setMessages((prev) => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: 'My agent is offline right now. You can still explore the wallet cards below or reach me at bg2896@nyu.edu.',
        error: true,
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="relative h-[100svh] w-full overflow-hidden bg-app text-ink font-sans selection:bg-accent/25">
      <BackgroundAmbient />

      {/* HOME + THEME — top-right chrome */}
      <div className="absolute top-6 right-6 z-50 flex items-center gap-2">
        <ThemeToggle />
        <a
          href="/"
          className="flex items-center justify-center w-10 h-10 rounded-lg bg-surface/65 backdrop-blur-md border border-accent/12 text-ink hover:bg-accent/12 hover:border-accent/25 transition-all"
          title="Go to home"
        >
          ⌂
        </a>
      </div>

      <div className="relative z-10 flex h-full flex-col">
        <div className="basis-[40%] min-h-0">
          <WalletChatPanel
            messages={messages}
            isTyping={isTyping}
            input={input}
            setInput={setInput}
            onSend={handleSend}
            suggestionChips={suggestionChips}
            chatBottomRef={chatBottomRef}
            activeCardLabel={activeCard ? CARD_MAP[activeCard]?.title : null}
          />
        </div>

        <div className="basis-[60%] min-h-0">
          <WalletStack
            cards={WALLET_CARD_DEFS}
            activeCard={activeCard}
            highlightedCard={highlightedCard}
            onSelectCard={handleCardSelect}
            onClearActiveCard={clearActiveCard}
          />
        </div>
      </div>
    </div>
  );
};

const WalletHome = () => {
  const [searchParams] = useSearchParams();
  const routeCard = searchParams.get('card');
  const initialCard = isWalletCardId(routeCard) ? routeCard : null;

  return (
    <WalletProvider initialCard={initialCard}>
      <WalletHomeScreen />
    </WalletProvider>
  );
};

export default WalletHome;
