import React, { useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { Globe, BookOpen, StopCircle, X } from 'lucide-react';
import PandaAvatar from './PandaAvatar';
import { useV2Stack } from '../../context/V2StackContext';

/**
 * Chat bar with three "modes" (collapsed / expanded / reply). Two visual
 * pieces — the panda avatar and the primary input slot — are shared across
 * all three modes via Framer Motion `layoutId`, so they MORPH between
 * positions instead of fading in and out:
 *
 *   collapsed                       expanded                       reply
 *   ─────────────────────────       ─────────────────────────      ────────────────────────
 *   [panda]  [───────input───]      [panda] bhavesh.ai             [panda]  last reply…
 *                                   ──────── transcript ───────
 *                                   [chips] [chips] [chips]
 *                                   [────── input ──────] [↑]
 *
 * The panda flies between corner positions; the input grows from a slim
 * pill to a full-width chat input. Other content (header, transcript,
 * chips, send button) crossfades around them via AnimatePresence.
 *
 * Streaming choreography (option H-1): isStreaming forces expanded,
 * tokens land in the live bot bubble, on `final` the chat collapses to
 * `reply` showing the last sentence.
 */

const STARTERS = [
  'who are you?',
  'tell me about TCS',
  'what is VoiceGraph?',
  'show me your design work',
  'how do i reach you?',
];

const CONTEXTUAL_STARTERS = {
  about: ['what are you focused on right now?', 'what makes you different?'],
  experience: ['what did you do at TCS?', 'tell me about Synopsys', 'your eSec internship?'],
  projects: ['which uses AI?', 'tell me about VoiceGraph', 'what did you ship most recently?'],
  publications: ['summarize your paper', 'what dataset?'],
  skills: ['your strongest stack?', 'mobile or backend?'],
  contact: ['email?', 'are you available?'],
  design: ['what tools do you use?'],
  leadership: ['what did you lead at GDSC?'],
  journey: ['why did you switch to ML?'],
  now: ['what are you shipping today?'],
};

const TOOL_ICON = { search_web: Globe, search_profile: BookOpen };

// Softer spring — the panda + input morph between modes should feel like
// physical objects settling into place, not snapping.
const SPRING = { type: 'spring', stiffness: 220, damping: 28, mass: 0.95 };

const ToolCallChip = ({ call }) => {
  const Icon = TOOL_ICON[call.name] || BookOpen;
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -4, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -4, scale: 0.95 }}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 10px',
        background: 'var(--v2-chip-bg)',
        border: '1px solid var(--v2-chip-border)',
        borderRadius: 999,
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 9,
        letterSpacing: 1.2,
        textTransform: 'uppercase',
        color: 'var(--v2-text-soft)',
      }}
    >
      <Icon size={10} />
      <span style={{ fontWeight: 600, color: 'var(--v2-accent)' }}>
        {call.name === 'search_web' ? 'web' : call.role || 'profile'}
      </span>
      {call.query && (
        <span
          style={{
            maxWidth: 180,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            color: 'var(--v2-text-softer)',
            textTransform: 'none',
            letterSpacing: 0,
          }}
        >
          {call.query}
        </span>
      )}
    </motion.div>
  );
};

const Bubble = ({ msg, prev }) => {
  const isUser = msg.role === 'user';
  const grouped = prev && prev.role === msg.role;
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginTop: grouped ? 3 : 8,
        marginBottom: 2,
      }}
    >
      <div
        style={{
          maxWidth: '78%',
          padding: '8px 12px',
          borderRadius: isUser
            ? `16px 16px ${grouped ? 16 : 4}px 16px`
            : `16px 16px 16px ${grouped ? 16 : 4}px`,
          background: isUser
            ? 'var(--v2-send-grad)'
            : msg.error
              ? 'var(--v2-error-bg)'
              : 'var(--v2-chip-bg)',
          color: isUser ? 'var(--v2-send-text)' : msg.error ? 'var(--v2-error-text)' : 'var(--v2-ink)',
          fontFamily: "'Inter Tight', sans-serif",
          fontSize: 13,
          lineHeight: 1.45,
          border: msg.error ? '1px solid var(--v2-error-border)' : 'none',
        }}
      >
        {msg.text || (msg.streaming ? <span style={{ opacity: 0.5 }}>…</span> : null)}
        {msg.streaming && (
          <span
            aria-hidden
            style={{
              display: 'inline-block',
              marginLeft: 2,
              width: 2,
              height: '0.95em',
              verticalAlign: 'middle',
              background: 'var(--v2-accent)',
              animation: 'blink 1s step-end infinite',
            }}
          />
        )}
      </div>
    </motion.div>
  );
};

const ChatBar = ({
  mode,                 // 'collapsed' | 'expanded' | 'reply'
  isMobile,
  messages,
  isStreaming,
  input,
  setInput,
  onSend,
  onStop,
  onFocusInput,
  onBlurInput,
  onMouseEnter,
  onMouseLeave,
  onDismiss,            // explicit close — used by ✕ button + Esc + outside click
}) => {
  const { agentStage, toolCalls, lastAgentMsg, dismissReply, activeCardId } = useV2Stack();
  const scrollRef = useRef(null);

  useEffect(() => {
    if (mode !== 'expanded') return;
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, mode]);

  const starters = useMemo(() => {
    return CONTEXTUAL_STARTERS[activeCardId] || STARTERS.slice(0, 3);
  }, [activeCardId]);

  const submit = (text) => {
    const v = (typeof text === 'string' ? text : input).trim();
    if (!v || isStreaming) return;
    onSend?.(v);
  };

  // Panda size per mode — driven by layoutId morph.
  const pandaSize = mode === 'expanded' ? 42 : 34;

  // Common bag passed to all three modes.
  const handlers = { onMouseEnter, onMouseLeave };
  const inputCommon = {
    value: input,
    onChange: (e) => setInput(e.target.value),
    onFocus: onFocusInput,
    onBlur: onBlurInput,
    placeholder: isStreaming ? 'streaming…' : 'ask bhavesh anything…',
    'aria-label': 'ask bhavesh',
    disabled: isStreaming,
  };

  return (
    <LayoutGroup id="v2-chat">
      <AnimatePresence mode="popLayout" initial={false}>
        {/* ── REPLY MODE ─────────────────────────────────────────────── */}
        {mode === 'reply' && lastAgentMsg && (
          <motion.div
            key="reply"
            {...handlers}
            onClick={dismissReply}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              padding: '0 28px',
              gap: 14,
              maxWidth: 760,
              margin: '0 auto',
              cursor: 'pointer',
              fontFamily: "'Inter Tight', sans-serif",
            }}
          >
            <motion.div layoutId="v2-panda" transition={SPRING}>
              <PandaAvatar stage={agentStage} size={32} />
            </motion.div>
            <motion.div
              layoutId="v2-input-slot"
              transition={SPRING}
              style={{
                flex: 1,
                fontSize: 14,
                color: 'var(--v2-ink)',
                lineHeight: 1.4,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                background: 'transparent',
                border: 'none',
                padding: 0,
              }}
            >
              {lastAgentMsg}
            </motion.div>
            <div
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
                letterSpacing: 1.5,
                color: 'var(--v2-placeholder)',
                textTransform: 'uppercase',
                whiteSpace: 'nowrap',
              }}
            >
              tap to dismiss
            </div>
          </motion.div>
        )}

        {/* ── COLLAPSED MODE ──────────────────────────────────────────── */}
        {mode === 'collapsed' && (
          <motion.div
            key="collapsed"
            {...handlers}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              padding: '0 24px',
              gap: 12,
              maxWidth: 760,
              margin: '0 auto',
              fontFamily: "'Inter Tight', sans-serif",
            }}
          >
            <motion.div layoutId="v2-panda" transition={SPRING}>
              <PandaAvatar stage={agentStage} size={32} />
            </motion.div>
            <motion.form
              onSubmit={(e) => { e.preventDefault(); submit(); }}
              style={{ flex: 1, display: 'flex', gap: 8 }}
            >
              <motion.input
                layoutId="v2-input-slot"
                transition={SPRING}
                {...inputCommon}
                style={{
                  flex: 1,
                  background: 'var(--v2-chip-bg)',
                  border: '1px solid var(--v2-chip-bg-strong)',
                  borderRadius: 999,
                  padding: '8px 16px',
                  color: 'var(--v2-ink)',
                  fontFamily: "'Inter Tight', sans-serif",
                  fontSize: 13,
                  outline: 'none',
                }}
              />
              {!isMobile && (
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 9,
                    letterSpacing: 1.5,
                    color: 'var(--v2-placeholder)',
                    textTransform: 'uppercase',
                    alignSelf: 'center',
                    whiteSpace: 'nowrap',
                  }}
                >
                  hover to expand
                </div>
              )}
            </motion.form>
          </motion.div>
        )}

        {/* ── EXPANDED MODE ───────────────────────────────────────────── */}
        {mode === 'expanded' && (
          <motion.div
            key="expanded"
            {...handlers}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              maxWidth: 640,
              margin: '0 auto',
              padding: '18px 24px 0',
              fontFamily: "'Inter Tight', sans-serif",
            }}
          >
            {/* header — panda + identity + close. The close button is the
                explicit escape hatch the user needs when they're stuck
                hovering the panel. Esc + outside-click also dismiss
                (wired in V2Home). */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                paddingBottom: 14,
                borderBottom: '1px solid var(--v2-chat-border)',
              }}
            >
              <motion.div layoutId="v2-panda" transition={SPRING}>
                <PandaAvatar stage={agentStage} size={pandaSize} />
              </motion.div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--v2-ink)' }}>bhavesh.ai</div>
                <div
                  style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 9,
                    letterSpacing: 1,
                    color: 'var(--v2-ink-mute)',
                  }}
                >
                  TALKS LIKE ME · {isStreaming ? 'THINKING…' : 'ESC OR TAP BELOW TO CLOSE'}
                </div>
              </div>
              {!isStreaming && onDismiss && (
                <button
                  type="button"
                  onClick={onDismiss}
                  aria-label="close chat"
                  title="close (Esc)"
                  style={{
                    flexShrink: 0,
                    width: 32,
                    height: 32,
                    borderRadius: '50%',
                    background: 'var(--v2-chip-bg)',
                    border: '1px solid var(--v2-chip-border-soft)',
                    color: 'var(--v2-ink-mute)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 200ms ease, color 200ms ease',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--v2-ink)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--v2-ink-mute)'; }}
                >
                  <X size={14} />
                </button>
              )}
            </div>

            {/* transcript */}
            <div
              ref={scrollRef}
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '14px 0',
                minHeight: 0,
              }}
            >
              {messages.length === 0 && (
                <div
                  style={{
                    padding: '20px 0',
                    fontSize: 13,
                    color: 'var(--v2-text-softer)',
                    fontStyle: 'italic',
                  }}
                >
                  ask anything — i&apos;ll open the right card as i answer.
                </div>
              )}
              {messages.map((m, i) => (
                <Bubble key={m.id} msg={m} prev={messages[i - 1]} />
              ))}

              {isStreaming && toolCalls.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                  <AnimatePresence initial={false}>
                    {toolCalls.map((tc) => (
                      <ToolCallChip key={tc.id} call={tc} />
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>

            {/* starter chips */}
            {!isStreaming && (
              <div
                style={{
                  display: 'flex',
                  gap: 6,
                  flexWrap: 'wrap',
                  padding: '0 0 10px',
                }}
              >
                {starters.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => submit(q)}
                    style={{
                      background: 'var(--v2-chip-bg)',
                      border: '1px solid var(--v2-chip-border-soft)',
                      color: 'var(--v2-text-soft)',
                      padding: '5px 12px',
                      borderRadius: 999,
                      fontSize: 11,
                      cursor: 'pointer',
                      fontFamily: "'Inter Tight', sans-serif",
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}

            <motion.form
              onSubmit={(e) => { e.preventDefault(); submit(); }}
              style={{
                display: 'flex',
                gap: 8,
                padding: '10px 0 16px',
                borderTop: '1px solid var(--v2-chat-border)',
              }}
            >
              <motion.input
                layoutId="v2-input-slot"
                transition={SPRING}
                {...inputCommon}
                style={{
                  flex: 1,
                  background: 'var(--v2-chip-bg)',
                  border: '1px solid var(--v2-chip-bg-strong)',
                  borderRadius: 999,
                  padding: '9px 16px',
                  color: 'var(--v2-ink)',
                  fontFamily: "'Inter Tight', sans-serif",
                  fontSize: 13,
                  outline: 'none',
                }}
              />
              {isStreaming ? (
                <button
                  type="button"
                  onClick={onStop}
                  style={{
                    background: 'var(--v2-stop-grad)',
                    border: 'none',
                    width: 36,
                    height: 36,
                    borderRadius: '50%',
                    color: 'var(--v2-stop-text)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  aria-label="stop streaming"
                >
                  <StopCircle size={15} />
                </button>
              ) : (
                <button
                  type="submit"
                  style={{
                    background: input.trim() ? 'var(--v2-send-grad)' : 'var(--v2-chip-bg)',
                    border: 'none',
                    width: 36,
                    height: 36,
                    borderRadius: '50%',
                    color: input.trim() ? 'var(--v2-send-text)' : 'var(--v2-placeholder)',
                    cursor: input.trim() ? 'pointer' : 'default',
                    fontSize: 16,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  aria-label="send"
                >
                  ↑
                </button>
              )}
            </motion.form>
          </motion.div>
        )}
      </AnimatePresence>
    </LayoutGroup>
  );
};

export default ChatBar;
