import React, {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { V2StackProvider, useV2Stack } from '../context/V2StackContext';
import CardStack from '../components/v2/CardStack';
import ChatBar from '../components/v2/ChatBar';
import AmbientBackground from '../components/v2/AmbientBackground';
import {
  IdentityPill,
  AvailabilityPill,
  ScrollHint,
  ProgressDots,
  TopRightChrome,
} from '../components/v2/Chrome';
import {
  resolveStackRoute,
  buildStackHref,
  getDeck,
} from '../components/v2/decks';
import { streamChat, StreamAbortError } from '../lib/streamClient';

const API_URL = import.meta.env.VITE_AGENT_API || 'http://localhost:8000';
const SESSION_KEY = 'v2.chat.v2'; // bumped — clear stale lab/v1 transcripts

const loadSession = () => {
  if (typeof window === 'undefined') return { messages: [], threadId: null };
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return { messages: [], threadId: null };
    const p = JSON.parse(raw);
    return {
      messages: Array.isArray(p.messages) ? p.messages : [],
      threadId: typeof p.threadId === 'string' ? p.threadId : null,
    };
  } catch { return { messages: [], threadId: null }; }
};

const persistSession = (messages, threadId) => {
  if (typeof window === 'undefined') return;
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify({ messages, threadId })); }
  catch { /* quota — ignore */ }
};

const V2HomeInner = () => {
  const location = useLocation();
  const navigateRouter = useNavigate();
  const {
    activeDeckId,
    activeCardId,
    openMode,
    applyFocusCommand,
    setAgentStage,
    pushToolCall,
    resetTurn,
    setFromRoute,
    lastAgentMsg,
    setLastAgentMsg,
    dismissReply,
    clearBuffering,
  } = useV2Stack();

  // URL → state on every location change.
  useEffect(() => {
    const suffix = location.pathname.replace(/^\/v2\/?/, '');
    const parsed = resolveStackRoute(suffix);
    setFromRoute(parsed);
  }, [location.pathname, setFromRoute]);

  // Lock body scroll while /v2 is mounted — kills the "page bounces a
  // little when scrolling" issue. Restored on unmount.
  useEffect(() => {
    document.documentElement.classList.add('v2-locked');
    document.body.classList.add('v2-locked');
    return () => {
      document.documentElement.classList.remove('v2-locked');
      document.body.classList.remove('v2-locked');
    };
  }, []);

  // Viewport tracking for mobile.
  const [vw, setVw] = useState(typeof window !== 'undefined' ? window.innerWidth : 1200);
  const [vh, setVh] = useState(typeof window !== 'undefined' ? window.innerHeight : 800);
  useEffect(() => {
    const onResize = () => { setVw(window.innerWidth); setVh(window.innerHeight); };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  const isMobile = vw < 640;
  // Tight viewport = short phone (e.g. iPhone SE landscape, small Android).
  // Below this we cap chat-expanded height so the stack zone has room.
  const isTightVH = vh < 640;

  // Session-restored chat state.
  const session = useRef(loadSession());
  const [messages, setMessages] = useState(session.current.messages);
  const [threadId, setThreadId] = useState(session.current.threadId);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingId, setStreamingId] = useState(null);
  const streamAbortRef = useRef(null);

  // Chat-bar mode logic. Streaming forces expanded; after a final lands the
  // bar collapses to a reply strip; otherwise hover/focus drive expansion.
  // `dismissed` is a hard override the user can set via ✕ / Esc / clicking
  // outside the chat panel — it short-circuits hovering+focusing so the
  // user can always escape an expanded chat that's covering the stack.
  const [hovering, setHovering] = useState(false);
  const [focusing, setFocusing] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  // Clear the dismiss flag whenever the user actively re-engages — sending
  // a question, mouse leaves entirely and re-enters fresh, etc. Otherwise
  // the dismiss would stick forever even if they click the input later.
  const handleDismissChat = useCallback(() => {
    setDismissed(true);
    setHovering(false);
    setFocusing(false);
    // Blur whatever's focused so the input stops claiming focusing=true.
    if (typeof document !== 'undefined') document.activeElement?.blur?.();
  }, []);

  const chatMode = useMemo(() => {
    if (isStreaming) return 'expanded';
    if (openMode && lastAgentMsg && !dismissed) return 'reply';
    if ((hovering || focusing) && !dismissed) return 'expanded';
    return 'collapsed';
  }, [isStreaming, openMode, lastAgentMsg, hovering, focusing, dismissed]);

  // Esc dismisses the expanded chat (unless an input is focused — then it
  // first blurs the input). Also resets dismiss when chat collapses back
  // naturally so the next hover behaves normally.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      if (chatMode === 'expanded' || chatMode === 'reply') {
        const tag = (document.activeElement?.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') {
          document.activeElement.blur();
          return;
        }
        e.preventDefault();
        handleDismissChat();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [chatMode, handleDismissChat]);

  // Auto-clear `dismissed` whenever the user successfully types something
  // new or sends a message — fresh engagement should re-enable hover.
  useEffect(() => {
    if (input.length > 0 && dismissed) setDismissed(false);
  }, [input, dismissed]);

  // Bar height by mode (matches mockup percentages).
  // On tight mobile viewports we cap expanded mode so the card stack below
  // keeps a usable ~45% slice — otherwise the focused card overflows or
  // peek tabs disappear.
  const chatHeightPct = useMemo(() => {
    if (chatMode === 'reply') return isMobile ? 18 : 14;
    if (chatMode === 'expanded') {
      if (isMobile) return isTightVH ? 55 : 65;
      return 50;
    }
    return isMobile ? 12 : 10;
  }, [chatMode, isMobile, isTightVH]);

  useEffect(() => {
    persistSession(messages, threadId);
  }, [messages, threadId]);

  // ── nav helpers ────────────────────────────────────────────────────────
  const goTo = useCallback((target, { replace = false } = {}) => {
    navigateRouter(buildStackHref(target), { replace });
  }, [navigateRouter]);

  // Wire the in-context navigate() to the router.
  // (V2StackContext calls onNavigate for every external request; we re-feed
  //  the URL so React Router stays the source of truth.)
  useEffect(() => {
    // Patch the context's navigate target side-effect by listening for
    // its calls — but applyFocusCommand and step calls go through the
    // context's own navigate() which doesn't have onNavigate plumbed in
    // by default. Provide it via the Provider's onNavigate prop in the
    // outer component instead.
  }, []);

  // ── streaming handler ───────────────────────────────────────────────────
  const handleStreamEvent = useCallback((event, { botMsgId }) => {
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
            const applied = applyFocusCommand(event.payload || {});
            if (applied) {
              const p = event.payload || {};
              goTo({ deckId: 'home', cardId: p.cardId, open: true });
            }
          });
        }
        break;
      case 'text_delta':
        clearBuffering(); // first text means the focused card is no longer "loading"
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
        clearBuffering();
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
        // Set the reply strip — H-1: after final, chat collapses to reply
        // showing the last sentence + the card stays open from the
        // ui_command. setLastAgentMsg here so chatMode flips to 'reply'.
        setLastAgentMsg(event.answer || '');
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
      default:
        break;
    }
  }, [applyFocusCommand, goTo, pushToolCall, setAgentStage, setLastAgentMsg]);

  const handleSend = useCallback(async (text = input) => {
    if (isStreaming) return;
    const q = (text || '').trim();
    if (!q) return;

    if (q.toLowerCase() === 'clear') {
      setMessages([]);
      setInput('');
      resetTurn();
      dismissReply();
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
    setStreamingId(botMsgId);
    resetTurn();
    setAgentStage('thinking');
    setIsStreaming(true);
    dismissReply(); // clear any prior reply-strip while we go fetch a new answer

    const controller = new AbortController();
    streamAbortRef.current = controller;

    try {
      await streamChat({
        apiUrl: API_URL,
        question: q,
        threadId,
        signal: controller.signal,
        onEvent: (event) => handleStreamEvent(event, { botMsgId }),
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
              text: 'My agent is offline right now. You can still scroll the cards below.',
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
      setStreamingId(null);
    }
  }, [dismissReply, handleStreamEvent, input, isStreaming, resetTurn, setAgentStage, threadId]);

  const handleStop = useCallback(() => {
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
  }, []);

  return (
    <div
      style={{
        position: 'relative',
        width: '100vw',
        height: '100svh',
        overflow: 'hidden',
        background: 'var(--v2-bg)',
        color: 'var(--v2-ink)',
        fontFamily: "'Inter Tight', sans-serif",
        transition: 'background 400ms ease, color 400ms ease',
      }}
    >
      {/* AmbientBackground = brand watermark + 3 mouse-parallax orbs +
          per-card tint crossfade. Lives behind everything. */}
      <AmbientBackground />

      {/* CHAT — pinned top, height varies by mode */}
      <div
        onMouseEnter={isMobile ? undefined : () => setHovering(true)}
        onMouseLeave={isMobile ? undefined : () => setHovering(false)}
        onClick={isMobile && chatMode === 'collapsed' ? () => setHovering(true) : undefined}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: `${chatHeightPct}%`,
          zIndex: 100,
          transition: 'height 0.52s cubic-bezier(0.32, 0.72, 0, 1), background 0.32s ease',
          background:
            chatMode === 'collapsed'
              ? 'linear-gradient(180deg, var(--v2-chat-bg-from), var(--v2-chat-bg-to))'
              : 'var(--v2-chat-bg-open)',
          backdropFilter: 'blur(14px)',
          borderBottom: '1px solid var(--v2-chat-border)',
        }}
      >
        <ChatBar
          mode={chatMode}
          isMobile={isMobile}
          messages={messages}
          isStreaming={isStreaming}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          onStop={handleStop}
          onFocusInput={() => { setFocusing(true); setDismissed(false); }}
          onBlurInput={() => {
            setFocusing(false);
            if (isMobile) setHovering(false);
          }}
          onDismiss={handleDismissChat}
        />
      </div>

      {/* STACK — fills below chat. A pointerdown on this region while the
          chat is expanded (and not streaming) dismisses the chat — gives
          the user an instant escape hatch by clicking anywhere on the
          deck. We only treat it as a dismiss when the chat is currently
          taking up its full expanded slot. */}
      <div
        onPointerDownCapture={() => {
          if (chatMode === 'expanded' && !isStreaming) handleDismissChat();
        }}
        style={{
          position: 'absolute',
          top: `${chatHeightPct}%`,
          left: 0,
          right: 0,
          bottom: 0,
          transition: 'top 0.52s cubic-bezier(0.32, 0.72, 0, 1)',
        }}
      >
        <CardStack />
      </div>

      {/* CHROME — pills + dots + hint, hidden when card is focused */}
      {!openMode && (
        <>
          <IdentityPill isMobile={isMobile} />
          <AvailabilityPill isMobile={isMobile} />
          {!isMobile && <ProgressDots />}
          {!activeCardId && <ScrollHint isMobile={isMobile} />}
        </>
      )}

      {/* HOME BUTTON + THEME TOGGLE — always visible top-right, above chat bar */}
      <TopRightChrome isMobile={isMobile} />
    </div>
  );
};

const V2Home = () => {
  const navigateRouter = useNavigate();
  const onNavigate = useCallback((target) => {
    navigateRouter(buildStackHref(target));
  }, [navigateRouter]);
  return (
    <V2StackProvider onNavigate={onNavigate}>
      <V2HomeInner />
    </V2StackProvider>
  );
};

export default V2Home;
