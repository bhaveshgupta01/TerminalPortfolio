import React, {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
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
import LabChatPanel from '../components/lab/LabChatPanel';
import LabStack from '../components/lab/LabStack';
import LabCommandPalette from '../components/lab/LabCommandPalette';
import { LAB_CARDS, LAB_CARD_BY_ID } from '../components/lab/labContent';
import {
  LabWalletProvider,
  isLabCardId,
  useLabWallet,
} from '../context/LabWalletContext';
import { streamChat, StreamAbortError } from '../lib/streamClient';

const API_URL = import.meta.env.VITE_AGENT_API || 'http://localhost:8000';

const SESSION_KEY = 'lab.chat.v1';

const suggestionChips = [
  { label: 'About', cmd: 'Who are you?', icon: <User size={14} /> },
  { label: 'Experience', cmd: 'Walk me through your work experience.', icon: <Briefcase size={14} /> },
  { label: 'Projects', cmd: 'What are your favorite projects?', icon: <Folder size={14} /> },
  { label: 'Tech Stack', cmd: 'What is your tech stack?', icon: <Cpu size={14} /> },
  { label: 'Design', cmd: 'Show me your design work.', icon: <Palette size={14} /> },
  { label: 'Publications', cmd: 'Tell me about your research paper.', icon: <BookOpen size={14} /> },
  { label: 'Contact', cmd: 'How do I contact you?', icon: <Mail size={14} /> },
  { label: 'Clear', cmd: 'clear', icon: <X size={14} /> },
];

const loadSession = () => {
  if (typeof window === 'undefined') return { messages: [], threadId: null };
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return { messages: [], threadId: null };
    const parsed = JSON.parse(raw);
    return {
      messages: Array.isArray(parsed.messages) ? parsed.messages : [],
      threadId: typeof parsed.threadId === 'string' ? parsed.threadId : null,
    };
  } catch {
    return { messages: [], threadId: null };
  }
};

const persistSession = (messages, threadId) => {
  if (typeof window === 'undefined') return;
  try {
    sessionStorage.setItem(
      SESSION_KEY,
      JSON.stringify({ messages, threadId }),
    );
  } catch { /* quota exceeded — non-fatal */ }
};

const LabHomeScreen = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    activeCard,
    activeCardSource,
    focusedSection,
    highlightPulseKey,
    setActiveCard,
    clearActiveCard,
    applyFocusCommand,
    setAgentStage,
    pushToolCall,
    resetTurn,
  } = useLabWallet();

  const restored = useRef(loadSession());
  const [messages, setMessages] = useState(restored.current.messages);
  const [threadId, setThreadId] = useState(restored.current.threadId);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const chatBottomRef = useRef(null);
  const streamAbortRef = useRef(null);

  // Persist chat as it changes.
  useEffect(() => {
    persistSession(messages, threadId);
  }, [messages, threadId]);

  // Auto-scroll chat.
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  // Keep the URL query in sync with the card state.
  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (activeCard) next.set('card', activeCard); else next.delete('card');
    if (activeCard && focusedSection) next.set('section', focusedSection); else next.delete('section');
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [activeCard, focusedSection, searchParams, setSearchParams]);

  // Global hotkeys: Cmd/Ctrl+K opens palette; Esc closes active card
  // (when no other handler is in the way); ←/→ when idle moves through cards.
  useEffect(() => {
    const onKey = (e) => {
      const isMeta = e.metaKey || e.ctrlKey;
      if (isMeta && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        return;
      }
      if (e.key === 'Escape' && activeCard && !paletteOpen) {
        const tag = (document.activeElement?.tagName || '').toLowerCase();
        if (tag !== 'input' && tag !== 'textarea') {
          clearActiveCard();
        }
      }
      if (!activeCard && !paletteOpen) {
        const tag = (document.activeElement?.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
          e.preventDefault();
          const [first] = LAB_CARDS;
          setActiveCard(first.id, { source: 'user' });
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [activeCard, paletteOpen, clearActiveCard, setActiveCard]);

  const activeCardLabel = activeCard ? LAB_CARD_BY_ID[activeCard]?.title : null;

  const handleCardSelect = useCallback((cardId) => {
    setActiveCard(cardId, { source: 'user' });
  }, [setActiveCard]);

  const handlePaletteSelect = useCallback((entry) => {
    setPaletteOpen(false);
    setActiveCard(entry.cardId, {
      source: 'palette',
      sectionId: entry.sectionId || null,
    });
  }, [setActiveCard]);

  const handleStop = useCallback(() => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
  }, []);

  const handleBotStreamEvent = useCallback((event, { botMsgId }) => {
    switch (event.type) {
      case 'thread':
        if (event.thread_id) setThreadId(event.thread_id);
        break;
      case 'status':
        if (event.stage) setAgentStage(event.stage);
        break;
      case 'tool_call':
        pushToolCall({ name: event.name, query: event.query || '', role: event.role || null });
        break;
      case 'ui_command':
        if (event.action === 'SET_ACTIVE_CARD') {
          startTransition(() => {
            applyFocusCommand(event.payload || {}, { respectUserLock: true });
          });
        }
        break;
      case 'text_delta':
        setMessages((prev) => prev.map((m) =>
          m.id === botMsgId
            ? { ...m, text: (m.text || '') + (event.content || ''), streaming: true }
            : m,
        ));
        break;
      case 'component':
        setMessages((prev) => prev.map((m) =>
          m.id === botMsgId ? { ...m, component: event.data || null } : m,
        ));
        break;
      case 'final':
        setMessages((prev) => prev.map((m) =>
          m.id === botMsgId
            ? {
              ...m,
              text: event.answer || m.text,
              source: event.source || m.source || 'profile',
              streaming: false,
            }
            : m,
        ));
        break;
      case 'error':
        setMessages((prev) => prev.map((m) =>
          m.id === botMsgId
            ? { ...m, text: event.message || 'Something went wrong.', error: true, streaming: false }
            : m,
        ));
        setAgentStage('error');
        break;
      case 'done':
        // handled by promise resolution
        break;
      default:
        break;
    }
  }, [applyFocusCommand, pushToolCall, setAgentStage]);

  const handleSend = useCallback(async (text = input) => {
    if (isStreaming) return;
    const q = (text || '').trim();
    if (!q) return;

    if (q.toLowerCase() === 'clear') {
      setMessages([]);
      setInput('');
      clearActiveCard();
      resetTurn();
      sessionStorage.removeItem(SESSION_KEY);
      return;
    }

    const userMsgId = `u-${Date.now()}`;
    const botMsgId = `b-${Date.now() + 1}`;
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', text: q },
      { id: botMsgId, role: 'bot', text: '', streaming: true, source: 'profile' },
    ]);
    setInput('');
    resetTurn();
    setAgentStage('thinking');
    setIsStreaming(true);

    const controller = new AbortController();
    streamAbortRef.current = controller;

    try {
      await streamChat({
        apiUrl: API_URL,
        question: q,
        threadId,
        signal: controller.signal,
        onEvent: (event) => handleBotStreamEvent(event, { botMsgId }),
      });
      setAgentStage('done');
    } catch (err) {
      if (err instanceof StreamAbortError) {
        setMessages((prev) => prev.map((m) =>
          m.id === botMsgId ? { ...m, streaming: false, text: m.text || 'Stopped.' } : m,
        ));
        setAgentStage('idle');
      } else {
        setMessages((prev) => prev.map((m) =>
          m.id === botMsgId
            ? {
              ...m,
              text: 'My agent is offline right now. You can still explore the cards below or reach me at bg2896@nyu.edu.',
              error: true,
              streaming: false,
            }
            : m,
        ));
        setAgentStage('error');
      }
    } finally {
      streamAbortRef.current = null;
      setIsStreaming(false);
    }
  }, [clearActiveCard, handleBotStreamEvent, input, isStreaming, resetTurn, setAgentStage, threadId]);

  return (
    <div className="relative h-[100svh] w-full overflow-hidden bg-app text-ink font-sans selection:bg-accent/25">
      <BackgroundAmbient />

      {/* HOME BUTTON + THEME TOGGLE — top-right chrome */}
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
          <LabChatPanel
            messages={messages}
            isStreaming={isStreaming}
            onStop={handleStop}
            input={input}
            setInput={setInput}
            onSend={handleSend}
            suggestionChips={suggestionChips}
            chatBottomRef={chatBottomRef}
            activeCardLabel={activeCardLabel}
            activeCardSource={activeCardSource}
            onOpenPalette={() => setPaletteOpen(true)}
          />
        </div>

        <div className="basis-[60%] min-h-0">
          <LabStack
            cards={LAB_CARDS}
            activeCard={activeCard}
            highlightedCard={activeCardSource === 'ai' ? activeCard : null}
            focusedSection={focusedSection}
            highlightPulseKey={highlightPulseKey}
            onSelectCard={handleCardSelect}
            onClearActiveCard={clearActiveCard}
          />
        </div>
      </div>

      <LabCommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        cards={LAB_CARDS}
        onSelect={handlePaletteSelect}
      />
    </div>
  );
};

const LabHome = () => {
  const [searchParams] = useSearchParams();
  const routeCard = searchParams.get('card');
  const routeSection = searchParams.get('section');
  const initialCard = isLabCardId(routeCard) ? routeCard : null;
  const initialSection = useMemo(() => {
    if (!initialCard) return null;
    const card = LAB_CARD_BY_ID[initialCard];
    if (!card) return null;
    const ok = (card.subsections || []).some((s) => s.id === routeSection);
    return ok ? routeSection : null;
  }, [initialCard, routeSection]);

  return (
    <LabWalletProvider initialCard={initialCard} initialSection={initialSection}>
      <LabHomeScreen />
    </LabWalletProvider>
  );
};

export default LabHome;
