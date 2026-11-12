import React from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Sparkles,
  Globe,
  ArrowLeft,
  Layers3,
  Search,
  BookOpen,
  Command,
  StopCircle,
} from 'lucide-react';
import ThemeToggle from '../ThemeToggle';
import DynamicComponent from '../DynamicComponent';
import LabAvatar from './LabAvatar';
import { useLabWallet } from '../../context/LabWalletContext';

const StageLabel = {
  idle: 'Ready',
  thinking: 'Thinking',
  searching_profile: 'Reading my notes',
  searching_web: 'Searching the web',
  tool_done: 'Synthesizing',
  speaking: 'Speaking',
  done: 'Done',
  error: 'Hit an error',
};

const ToolCallRow = ({ call }) => {
  const Icon = call.name === 'search_web' ? Globe : BookOpen;
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 8 }}
      className="glass-chip inline-flex max-w-full items-center gap-2 rounded-full px-3 py-1.5 text-xs text-muted"
    >
      <Icon size={12} className="shrink-0" />
      <span className="font-semibold text-ink">
        {call.name === 'search_web' ? 'web' : call.role ? `profile · ${call.role}` : 'profile'}
      </span>
      {call.query && (
        <span className="min-w-0 truncate text-faint">{call.query}</span>
      )}
    </motion.div>
  );
};

const LabChatPanel = ({
  messages,
  isStreaming,
  onStop,
  input,
  setInput,
  onSend,
  suggestionChips,
  chatBottomRef,
  activeCardLabel,
  activeCardSource,
  onOpenPalette,
}) => {
  const { agentStage, toolCalls } = useLabWallet();

  const chatState = agentStage && agentStage !== 'idle'
    ? agentStage
    : activeCardLabel && activeCardSource === 'ai'
      ? 'done'
      : messages.length > 0
        ? 'done'
        : 'idle';

  return (
    <div className="h-full px-3 pt-3 md:px-6 md:pt-6">
      <div className="glass-strong glass-edge relative flex h-full flex-col overflow-hidden rounded-[32px] border border-subtle/15">
        <div className="absolute inset-0 bg-gradient-to-br from-rose-500/12 via-transparent to-amber-400/8" />
        <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-white/8 to-transparent dark:from-white/4" />

        <div className="relative z-10 flex items-center justify-between gap-3 border-b border-subtle/10 px-4 py-3 md:px-5">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-accent/20 bg-accent/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">
                Lab · streaming
              </span>
              {activeCardLabel && (
                <span className="rounded-full border border-subtle/15 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">
                  {activeCardSource === 'ai' ? 'AI focused on ' : 'Focused on '}
                  {activeCardLabel}
                </span>
              )}
              <span className="rounded-full border border-subtle/15 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">
                {StageLabel[agentStage] || 'Ready'}
              </span>
            </div>
            <h1 className="mt-2 text-lg font-bold tracking-tight text-ink md:text-xl">
              Ask, and the card you need comes up as I speak.
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onOpenPalette}
              className="glass-chip hidden items-center gap-2 rounded-full px-3 py-2 text-xs font-medium text-muted transition-colors hover:text-ink md:inline-flex"
              aria-label="Open command palette (Cmd+K)"
              title="Command palette — ⌘K"
            >
              <Command size={13} />
              <span>⌘K</span>
            </button>
            <Link
              to="/wallet"
              className="glass-chip hidden items-center gap-2 rounded-full px-3 py-2 text-xs font-medium text-muted transition-colors hover:text-ink md:inline-flex"
            >
              <ArrowLeft size={14} />
              Wallet v1
            </Link>
            <ThemeToggle />
          </div>
        </div>

        <div className="relative z-10 grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 sm:grid-cols-[auto_minmax(0,1fr)] md:px-5 md:pb-5">
          <div className="hidden shrink-0 sm:flex sm:flex-col sm:items-center sm:justify-between sm:py-2">
            <LabAvatar stage={chatState} />
            <div className="rounded-[24px] border border-subtle/15 bg-surface/60 px-4 py-3 text-center">
              <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-faint">
                Interface rule
              </div>
              <div className="mt-2 max-w-[10rem] text-xs leading-relaxed text-muted">
                The agent narrates; the card below reacts in the same frame.
              </div>
            </div>
          </div>

          <div className="flex min-h-0 flex-col">
            <div className="mb-3 flex items-center gap-3 sm:hidden">
              <LabAvatar stage={chatState} />
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-faint">
                  {StageLabel[chatState] || 'Ready'}
                </div>
                <p className="mt-1 text-sm leading-relaxed text-muted">
                  Streaming chat with live card focus and phrase highlighting.
                </p>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto pr-1 scrollbar-hide">
              {messages.length === 0 ? (
                <div className="flex h-full flex-col justify-center rounded-[28px] border border-dashed border-subtle/15 bg-surface/45 px-5 py-6">
                  <div className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <Layers3 size={16} className="text-accent" />
                    Try: &ldquo;tell me about TCS&rdquo;, &ldquo;what&rsquo;s VoiceGraph?&rdquo;, or &ldquo;contact&rdquo;
                  </div>
                  <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
                    Tokens stream in live. When the agent knows the answer is in a card, it focuses that card and pulses the exact phrases it&rsquo;s quoting.
                  </p>
                  <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px] text-faint">
                    <Search size={12} />
                    <span>Watch the chip tray — it shows each tool call live.</span>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {messages.map((msg) => (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[92%] rounded-[24px] px-4 py-3 ${
                          msg.role === 'user'
                            ? 'bg-ink text-app'
                            : msg.error
                              ? 'glass glass-tint-rose text-rose-300'
                              : 'glass glass-tint-rose text-ink'
                        }`}
                      >
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">
                          {msg.text}
                          {msg.role === 'bot' && msg.streaming && (
                            <span className="ml-0.5 inline-block h-[1em] w-[2px] align-middle bg-accent animate-blink" aria-hidden />
                          )}
                        </p>
                        {msg.component && (
                          <div className="mt-3">
                            <DynamicComponent component={msg.component} />
                          </div>
                        )}
                        {msg.role === 'bot' && !msg.streaming && (msg.source === 'web' || msg.source === 'both') && (
                          <div className="mt-3 flex items-center gap-1 text-[10px] text-faint">
                            <Globe size={10} />
                            {msg.source === 'web' ? 'answered from the web' : 'profile + web'}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}

                  {isStreaming && toolCalls.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      <AnimatePresence initial={false}>
                        {toolCalls.map((call) => (
                          <ToolCallRow key={call.id} call={call} />
                        ))}
                      </AnimatePresence>
                    </div>
                  )}

                  {isStreaming && agentStage === 'thinking' && (
                    <div className="flex justify-start">
                      <div className="glass glass-tint-rose flex items-center gap-1.5 rounded-[24px] px-4 py-3">
                        {[0, 1, 2].map((idx) => (
                          <motion.span
                            key={idx}
                            className="h-1.5 w-1.5 rounded-full bg-accent"
                            animate={{ y: [0, -3, 0] }}
                            transition={{ duration: 0.8, repeat: Infinity, delay: idx * 0.12 }}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  <div ref={chatBottomRef} />
                </div>
              )}
            </div>

            <div className="mt-4 overflow-x-auto scrollbar-hide">
              <div className="flex gap-2 whitespace-nowrap">
                {suggestionChips.map((chip) => (
                  <button
                    key={chip.label}
                    type="button"
                    onClick={() => onSend(chip.cmd)}
                    className="glass-chip inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs font-medium text-muted transition-colors hover:text-ink"
                  >
                    {chip.icon}
                    {chip.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-3">
              <div className="flex items-center gap-2 rounded-[24px] border border-subtle/15 bg-surface/45 px-2 py-2 focus-within:border-accent/30">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-accent/10 text-accent">
                  <Sparkles size={18} />
                </div>
                <input
                  type="text"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !isStreaming) onSend();
                  }}
                  placeholder={isStreaming ? 'Streaming response…' : 'Ask about TCS, projects, contact, publications…'}
                  className="h-11 flex-1 bg-transparent px-1 text-sm text-ink outline-none placeholder:text-faint"
                  aria-label="Ask a question"
                  disabled={isStreaming}
                />
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={onStop}
                    className="flex h-11 w-11 items-center justify-center rounded-2xl bg-rose-500 text-white transition-opacity hover:opacity-90"
                    aria-label="Stop streaming"
                    title="Stop"
                  >
                    <StopCircle size={16} />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => onSend()}
                    className="flex h-11 w-11 items-center justify-center rounded-2xl bg-ink text-app transition-opacity hover:opacity-90"
                    aria-label="Send message"
                  >
                    <Send size={16} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LabChatPanel;
