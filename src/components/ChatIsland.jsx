import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Sparkles, ChevronUp, Globe, CornerDownLeft, X,
} from 'lucide-react';
import DynamicComponent from './DynamicComponent';

/**
 * Dynamic-Island-style chat surface.
 *
 * States:
 *   idle      → small pill ("Ask me anything…")
 *   expanded  → full chat (messages + chips + input)
 *
 * Triggers for expanded:
 *   - hovered (with 700ms leave debounce)
 *   - input focused
 *   - input has text
 *   - typing animation in progress
 *   - message just arrived (sticky for 5s)
 *
 * Adaptive opacity: the whole island dims when the user is scrolling
 *   the page AND not interacting with the chat.
 */
const SPRING = { type: 'spring', stiffness: 260, damping: 28, mass: 0.9 };

const ChatIsland = ({
  messages,
  isTyping,
  input,
  setInput,
  onSend,
  suggestionChips,
  isScrolling,
  chatBottomRef,
}) => {
  const [hoverBuffer, setHoverBuffer] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [recentMessage, setRecentMessage] = useState(false);
  const hoverTimer = useRef(null);
  const recentTimer = useRef(null);
  const inputRef = useRef(null);

  const hasMessages = messages.length > 0;

  const expanded =
    hoverBuffer ||
    isFocused ||
    isTyping ||
    recentMessage ||
    input.length > 0;

  // stickiness after bot reply arrives
  const lastBotId = hasMessages ? messages[messages.length - 1].id : null;
  useEffect(() => {
    if (!hasMessages) return;
    const last = messages[messages.length - 1];
    if (last.role !== 'bot') return;
    setRecentMessage(true);
    if (recentTimer.current) clearTimeout(recentTimer.current);
    recentTimer.current = setTimeout(() => setRecentMessage(false), 5000);
  }, [lastBotId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    if (recentTimer.current) clearTimeout(recentTimer.current);
  }, []);

  const handleMouseEnter = () => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    setHoverBuffer(true);
  };
  const handleMouseLeave = () => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = setTimeout(() => setHoverBuffer(false), 700);
  };

  const handlePillClick = () => {
    setHoverBuffer(true);
    requestAnimationFrame(() => inputRef.current?.focus());
  };

  const handleCollapse = (e) => {
    e?.stopPropagation();
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    setHoverBuffer(false);
    setIsFocused(false);
    setRecentMessage(false);
    inputRef.current?.blur();
  };

  const chatIsActive = expanded;
  const dimmed = isScrolling && !chatIsActive;

  // aura tint based on state
  const auraColor = isTyping
    ? 'rgba(251, 191, 36, 0.35)'     // amber while thinking
    : recentMessage
      ? 'rgba(16, 185, 129, 0.30)'   // emerald right after reply
      : expanded
        ? 'rgba(244, 63, 94, 0.30)'  // rose when active
        : 'rgba(244, 63, 94, 0.18)'; // quiet rose idle

  return (
    <div
      className="absolute inset-x-0 bottom-0 z-50 pointer-events-none flex justify-center px-4 md:px-6"
      style={{ paddingBottom: 'calc(env(safe-area-inset-bottom, 0px) + 1rem)' }}
    >
      <motion.div
        layout
        transition={SPRING}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        animate={{
          opacity: dimmed ? 0.32 : 1,
          y: dimmed ? 4 : 0,
          scale: dimmed ? 0.985 : 1,
          filter: dimmed ? 'blur(0.4px)' : 'blur(0px)',
        }}
        className={
          'pointer-events-auto relative ' +
          (expanded
            ? 'w-full max-w-3xl'
            : 'w-72 sm:w-80')
        }
      >
        {/* aura glow */}
        <motion.div
          aria-hidden
          className="absolute -inset-10 -z-10 rounded-[64px] blur-3xl pointer-events-none"
          animate={{
            background: `radial-gradient(ellipse at center, ${auraColor}, transparent 65%)`,
            scale: expanded ? 1.15 : 1,
            opacity: dimmed ? 0.2 : 1,
          }}
          transition={{ duration: 0.6 }}
        />

        {/* main island */}
        <motion.div
          layout
          transition={SPRING}
          className="glass-strong glass-edge relative overflow-hidden"
          animate={{
            borderRadius: expanded ? 28 : 999,
          }}
        >
          <AnimatePresence mode="wait" initial={false}>
            {expanded ? (
              <motion.div
                key="expanded"
                layout
                initial={{ opacity: 0, filter: 'blur(4px)' }}
                animate={{ opacity: 1, filter: 'blur(0px)' }}
                exit={{ opacity: 0, filter: 'blur(4px)' }}
                transition={{ duration: 0.22 }}
                className="flex flex-col p-2"
              >
                {/* collapse button */}
                {hasMessages && (
                  <button
                    onClick={handleCollapse}
                    aria-label="Collapse chat"
                    className="absolute top-3 right-3 z-20 p-1.5 rounded-full glass-chip text-muted hover:text-ink"
                  >
                    <X size={14} />
                  </button>
                )}

                {/* messages */}
                <AnimatePresence initial={false}>
                  {hasMessages && (
                    <motion.div
                      key="messages"
                      layout
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.25 }}
                      className="w-full max-h-[58vh] overflow-y-auto scrollbar-hide px-2 pt-2 pb-2"
                    >
                      <div className="space-y-3">
                        {messages.map((msg) => (
                          <motion.div
                            key={msg.id}
                            layout
                            initial={{ opacity: 0, y: 12, scale: 0.98 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            transition={SPRING}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                          >
                            <div
                              className={`px-4 py-2.5 rounded-2xl max-w-[85%] glass ${
                                msg.role === 'user'
                                  ? 'glass-tint-neutral text-ink'
                                  : msg.error
                                    ? 'glass-tint-rose text-rose-400'
                                    : 'glass-tint-rose text-ink'
                              }`}
                            >
                              <p className="text-sm md:text-[15px] leading-relaxed whitespace-pre-wrap relative z-10">
                                {msg.text}
                              </p>
                              {msg.component && (
                                <div className="mt-3 relative z-10">
                                  <DynamicComponent component={msg.component} />
                                </div>
                              )}
                              {msg.role === 'bot' && (msg.source === 'web' || msg.source === 'both') && (
                                <div className="mt-2 flex items-center gap-1 text-[10px] text-faint relative z-10">
                                  <Globe size={10} />
                                  {msg.source === 'web' ? 'answered from the web' : 'profile + web'}
                                </div>
                              )}
                            </div>
                          </motion.div>
                        ))}

                        {isTyping && (
                          <motion.div
                            layout
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex justify-start"
                          >
                            <div className="glass glass-tint-rose px-4 py-2.5 rounded-2xl flex gap-1.5 items-center">
                              <motion.span
                                className="w-1.5 h-1.5 bg-rose-400 rounded-full"
                                animate={{ y: [0, -3, 0] }}
                                transition={{ duration: 0.8, repeat: Infinity, delay: 0 }}
                              />
                              <motion.span
                                className="w-1.5 h-1.5 bg-amber-400 rounded-full"
                                animate={{ y: [0, -3, 0] }}
                                transition={{ duration: 0.8, repeat: Infinity, delay: 0.12 }}
                              />
                              <motion.span
                                className="w-1.5 h-1.5 bg-emerald-400 rounded-full"
                                animate={{ y: [0, -3, 0] }}
                                transition={{ duration: 0.8, repeat: Infinity, delay: 0.24 }}
                              />
                            </div>
                          </motion.div>
                        )}
                        <div ref={chatBottomRef} />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* suggestion chips */}
                <motion.div
                  layout
                  className="w-full overflow-x-auto scrollbar-hide px-1 py-1"
                >
                  <div className="flex gap-2 whitespace-nowrap">
                    {suggestionChips.map((chip, idx) => (
                      <motion.button
                        key={idx}
                        whileHover={{ scale: 1.05, y: -1 }}
                        whileTap={{ scale: 0.96 }}
                        onClick={() => onSend(chip.cmd)}
                        className="glass-chip rounded-full flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-muted hover:text-ink transition-colors"
                      >
                        {chip.icon}
                        {chip.label}
                      </motion.button>
                    ))}
                  </div>
                </motion.div>

                {/* input row */}
                <motion.div layout className="w-full mt-1">
                  <div className="flex items-center gap-2 px-1.5 py-1.5 rounded-2xl bg-subtle/[0.04] ring-1 ring-inset ring-subtle/10 focus-within:ring-accent/30 transition-all">
                    <div className="p-2.5 rounded-xl bg-subtle/[0.06] text-accent relative z-10">
                      <Sparkles size={18} />
                    </div>
                    <input
                      ref={inputRef}
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') onSend(); if (e.key === 'Escape') handleCollapse(); }}
                      onFocus={() => setIsFocused(true)}
                      onBlur={() => setIsFocused(false)}
                      placeholder="Ask about my projects, TCS, hackathons, anything…"
                      className="flex-1 bg-transparent border-none outline-none text-ink placeholder-faint h-10 px-1 text-sm md:text-[15px] font-medium relative z-10"
                    />
                    <motion.button
                      whileHover={{ scale: 1.06 }}
                      whileTap={{ scale: 0.94 }}
                      onClick={() => onSend()}
                      className="p-2.5 bg-ink text-app rounded-xl shadow-lg shadow-accent/20 hover:opacity-90 relative z-10"
                      aria-label="Send message"
                    >
                      <Send size={16} />
                    </motion.button>
                  </div>
                </motion.div>
              </motion.div>
            ) : (
              <motion.button
                key="idle"
                layout
                initial={{ opacity: 0, filter: 'blur(4px)' }}
                animate={{ opacity: 1, filter: 'blur(0px)' }}
                exit={{ opacity: 0, filter: 'blur(4px)' }}
                transition={{ duration: 0.22 }}
                onClick={handlePillClick}
                className="w-full h-[52px] flex items-center justify-between px-4 cursor-pointer group"
              >
                <div className="flex items-center gap-2 relative z-10">
                  <motion.div
                    animate={{ rotate: [0, 8, -6, 0] }}
                    transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
                    className="text-accent"
                  >
                    <Sparkles size={16} />
                  </motion.div>
                  <span className="text-[13px] text-muted group-hover:text-ink transition-colors font-medium">
                    {recentMessage && hasMessages ? 'Reply received' : 'Ask me anything…'}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 relative z-10">
                  {hasMessages && (
                    <span className="text-[10px] text-faint font-mono">
                      {messages.length}
                    </span>
                  )}
                  <ChevronUp size={14} className="text-faint group-hover:text-muted transition-colors" />
                </div>
              </motion.button>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default ChatIsland;
