import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Send,
  Sparkles,
  Globe,
  ArrowLeft,
  ArrowRight,
  Layers3,
} from 'lucide-react';
import ThemeToggle from '../ThemeToggle';
import DynamicComponent from '../DynamicComponent';
import WalletAvatar from './WalletAvatar';

const WalletChatPanel = ({
  messages,
  isTyping,
  input,
  setInput,
  onSend,
  suggestionChips,
  chatBottomRef,
  activeCardLabel,
}) => {
  const chatState = isTyping
    ? 'thinking'
    : activeCardLabel
      ? 'guiding'
      : messages.length > 0
        ? 'replied'
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
                Wallet Mode
              </span>
              {activeCardLabel && (
                <span className="rounded-full border border-subtle/15 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">
                  Focused on {activeCardLabel}
                </span>
              )}
            </div>
            <h1 className="mt-2 text-lg font-bold tracking-tight text-ink md:text-xl">
              Chat island on top. Wallet stack below.
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <Link
              to="/"
              className="glass-chip hidden items-center gap-2 rounded-full px-3 py-2 text-xs font-medium text-muted transition-colors hover:text-ink md:inline-flex"
            >
              <ArrowLeft size={14} />
              Classic
            </Link>
            <Link
              to="/lab"
              className="glass-chip hidden items-center gap-2 rounded-full px-3 py-2 text-xs font-medium text-muted transition-colors hover:text-ink md:inline-flex"
              title="Try the streaming lab version"
            >
              Lab
              <ArrowRight size={14} />
            </Link>
            <ThemeToggle />
          </div>
        </div>

        <div className="relative z-10 grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 sm:grid-cols-[auto_minmax(0,1fr)] md:px-5 md:pb-5">
          <div className="hidden shrink-0 sm:flex sm:flex-col sm:items-center sm:justify-between sm:py-2">
            <WalletAvatar state={chatState} />
            <div className="rounded-[24px] border border-subtle/15 bg-surface/60 px-4 py-3 text-center">
              <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-faint">
                Interface rule
              </div>
              <div className="mt-2 max-w-[10rem] text-xs leading-relaxed text-muted">
                The AI explains on top while the wallet stack changes below.
              </div>
            </div>
          </div>

          <div className="flex min-h-0 flex-col">
            <div className="mb-3 flex items-center gap-3 sm:hidden">
              <WalletAvatar state={chatState} />
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-faint">
                  AI state
                </div>
                <p className="mt-1 text-sm leading-relaxed text-muted">
                  It can answer, render structured components, and focus a wallet card.
                </p>
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto pr-1 scrollbar-hide">
              {messages.length === 0 ? (
                <div className="flex h-full flex-col justify-center rounded-[28px] border border-dashed border-subtle/15 bg-surface/45 px-5 py-6">
                  <div className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <Layers3 size={16} className="text-accent" />
                    Ask about experience, projects, contact, or design
                  </div>
                  <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
                    This first pass still uses the existing backend JSON response. The next iteration can upgrade the same UI to partial text streaming and explicit `ui_command` events without changing the wallet state model.
                  </p>
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
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</p>
                        {msg.component && (
                          <div className="mt-3">
                            <DynamicComponent component={msg.component} />
                          </div>
                        )}
                        {msg.role === 'bot' && (msg.source === 'web' || msg.source === 'both') && (
                          <div className="mt-3 flex items-center gap-1 text-[10px] text-faint">
                            <Globe size={10} />
                            {msg.source === 'web' ? 'answered from the web' : 'profile + web'}
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}

                  {isTyping && (
                    <div className="flex justify-start">
                      <div className="glass glass-tint-rose flex items-center gap-1.5 rounded-[24px] px-4 py-3">
                        {[0, 1, 2].map((idx) => (
                          <motion.span
                            // eslint-disable-next-line react/no-array-index-key
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
                    if (event.key === 'Enter') onSend();
                  }}
                  placeholder="Ask about TCS, projects, contact, publications..."
                  className="h-11 flex-1 bg-transparent px-1 text-sm text-ink outline-none placeholder:text-faint"
                />
                <button
                  type="button"
                  onClick={() => onSend()}
                  className="flex h-11 w-11 items-center justify-center rounded-2xl bg-ink text-app transition-opacity hover:opacity-90"
                  aria-label="Send message"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WalletChatPanel;
