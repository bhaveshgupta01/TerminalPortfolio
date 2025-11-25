
import { 
  Terminal, Code, User, Briefcase, Mail, Cpu, X, Minus, Square, 
  Play, Folder, FileText, ChevronRight, Github, Linkedin, 
  Globe, Database, Server, Wifi, Zap
} from 'lucide-react';

import React, { useState, useEffect, useRef } from 'react';

const App = () => {
    const [input, setInput] = useState('');
  const [history, setHistory] = useState([
    { type: 'system', content: 'Initializing NYU_MSCS_KERNEL v3.0...' },
    { type: 'system', content: 'Mounting virtual file system...' },
    { type: 'success', content: 'CONNECTED. Welcome, Developer.' },
    { type: 'info', content: 'Type "help" to begin exploration.' },
  ]);
  const [activeTab, setActiveTab] = useState('terminal');
  const [isBooting, setIsBooting] = useState(true);
  const [time, setTime] = useState(new Date().toLocaleTimeString());
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
    const commandHistoryRef = useRef([]);
    const [historyIndex, setHistoryIndex] = useState(null);

  // Boot Sequence
  useEffect(() => {
    setTimeout(() => setIsBooting(false), 2500);
    const timer = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);

    // Load/save history to localStorage
    useEffect(() => {
        try {
            const saved = localStorage.getItem('terminal_history');
            if (saved) setHistory(JSON.parse(saved));
        } catch (e) {
            // ignore
        }
    }, []);

    useEffect(() => {
        try {
            localStorage.setItem('terminal_history', JSON.stringify(history));
        } catch (e) {
            // ignore
        }
    }, [history]);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  // Focus input on click
  const focusInput = () => {
    if (activeTab === 'terminal') inputRef.current?.focus();
  }

  const commands = {
    help: {
      text: "Available commands:",
      list: ["about", "stack", "projects", "contact", "clear", "run <lang>"]
    },
    about: "MSCS Student at NYU. Specializing in Distributed Systems & High-Performance Computing. 30 years mentor guidance.",
    stack: "Java (Spring/Micronaut), Python (FastAPI/Django), Kotlin, AWS, Docker, Kubernetes.",
    contact: "Email: student@nyu.edu | LinkedIn: /in/student",
    clear: "CLEAR_ACTION",
  };

  const projects = [
    {
      id: 'java-engine',
      name: 'TradeEngine.java',
      lang: 'java',
      icon: <Database size={16} className="text-orange-400"/>,
      desc: 'LMAX Disruptor based matching engine.',
      code: `public class OrderBook {
    private final Long2ObjectHashMap<Order> bids;
    private final Long2ObjectHashMap<Order> asks;

    public void process(Order order) {
        // O(1) lookup using primitive collections
        if (order.isLimit()) {
            matchLimitOrder(order);
        }
    }
}`
    },
    {
      id: 'py-ai',
      name: 'Transformer.py',
      lang: 'python',
      icon: <Zap size={16} className="text-yellow-300"/>,
      desc: 'Custom Attention Mechanism implementation.',
      code: `import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"`
    },
    {
      id: 'kt-safe',
      name: 'SecureNet.kt',
      lang: 'kotlin',
      icon: <Server size={16} className="text-purple-400"/>,
      desc: 'Encrypted P2P mesh network node.',
      code: `import io.ktor.network.selector.*
import io.ktor.network.sockets.*
import kotlinx.coroutines.*

fun main() = runBlocking {
    val selector = ActorSelectorManager(Dispatchers.IO)
    val server = aSocket(selector).tcp().bind("127.0.0.1", 9002)
    println("Secure Node Listening...")
}`
    }
  ];

    // Process a raw command string (extracted for reuse)
    const processCommand = (cmdRaw) => {
        const trimmed = (cmdRaw || '').trim();
        if (!trimmed) return;

        const args = trimmed.toLowerCase().split(' ');
        const mainCmd = args[0];

        // Add user command to history log and command history
        setHistory(prev => [...prev, { type: 'user', content: trimmed }]);
        commandHistoryRef.current.push(trimmed);
        setHistoryIndex(commandHistoryRef.current.length);
        setInput('');

        // Handle built-in commands
        if (mainCmd === 'clear') {
            setHistory([]);
            return;
        }

        if (commands[mainCmd]) {
            const res = commands[mainCmd];
            if (res.list) {
                setHistory(prev => [...prev, { type: 'info', content: res.text }, { type: 'list', content: res.list }]);
            } else {
                setHistory(prev => [...prev, { type: 'success', content: res }]);
            }
            return;
        }

        if (mainCmd === 'run') {
            const lang = args[1];
            const proj = projects.find(p => p.lang.includes(lang));

            if (proj) {
                setHistory(prev => [...prev, { type: 'info', content: `Compiling ${proj.name}...` }]);
                setTimeout(() => {
                    setHistory(prev => [...prev,
                        { type: 'success', content: `Build Successful (24ms)` },
                        { type: 'output', content: `> Executing artifact...` },
                        { type: 'output', content: `> [STDOUT] System initialized.` },
                        { type: 'output', content: `> [STDOUT] Listening on port 8080.` },
                    ]);
                }, 800);
            } else {
                setHistory(prev => [...prev, { type: 'error', content: `Error: Module '${lang}' not found. Try: java, python, kotlin` }]);
            }
            return;
        }

        setHistory(prev => [...prev, { type: 'error', content: `Command not found: ${mainCmd}` }]);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            processCommand(input);
            return;
        }

        if (e.key === 'ArrowUp') {
            e.preventDefault();
            const idx = historyIndex === null ? commandHistoryRef.current.length : historyIndex;
            const next = Math.max(0, (idx || 0) - 1);
            const cmd = commandHistoryRef.current[next] || '';
            setInput(cmd);
            setHistoryIndex(next);
            return;
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            const idx = historyIndex === null ? commandHistoryRef.current.length : historyIndex;
            const next = Math.min(commandHistoryRef.current.length, (idx || 0) + 1);
            const cmd = commandHistoryRef.current[next] || '';
            setInput(cmd);
            setHistoryIndex(next === commandHistoryRef.current.length ? null : next);
            return;
        }
    };

  if (isBooting) {
    return (
      <div className="h-screen w-full bg-black flex flex-col items-center justify-center font-mono text-green-500 p-4">
        <div className="w-full max-w-md space-y-2">
          <div className="flex justify-between text-xs text-gray-500">
            <span>BIOS v4.01</span>
            <span>MEM: 64GB OK</span>
          </div>
          <div className="h-1 w-full bg-gray-900 rounded overflow-hidden">
            <div className="h-full bg-green-500 animate-pulse w-3/4"></div>
          </div>
                    <div className="text-xs space-y-1">
                        <p>{'>'} LOADING KERNEL MODULES...</p>
                        <p>{'>'} MOUNTING FILESYSTEM [RW]...</p>
                        <p>{'>'} STARTING REACT RENDERER...</p>
                    </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full bg-[#0a0a0a] text-gray-300 font-mono flex flex-col overflow-hidden relative selection:bg-green-900 selection:text-green-100">
      
      {/* Scanline Effect Overlay */}
      <div className="scanline z-50 pointer-events-none"></div>

      {/* TOP BAR (Mac/Window Style) */}
      <div className="h-10 bg-[#111] border-b border-[#222] flex items-center justify-between px-4 z-10 select-none">
        <div className="flex items-center space-x-2">
            <div className="flex space-x-2 group">
                <div className="w-3 h-3 rounded-full bg-red-500/80 group-hover:bg-red-500 transition-colors"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500/80 group-hover:bg-yellow-500 transition-colors"></div>
                <div className="w-3 h-3 rounded-full bg-green-500/80 group-hover:bg-green-500 transition-colors"></div>
            </div>
            <span className="ml-4 text-xs text-gray-500 font-medium tracking-wide">dev@nyu-portfolio:~</span>
        </div>
        <div className="flex items-center space-x-2 text-gray-600">
            <Minus size={14} />
            <Square size={12} />
            <X size={14} />
        </div>
      </div>

      {/* MAIN LAYOUT */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* SIDEBAR (Explorer) */}
        <div className="w-16 md:w-64 bg-[#0d0d0d] border-r border-[#222] flex flex-col hidden md:flex">
            <div className="p-3 text-xs font-bold text-gray-500 tracking-widest uppercase">Explorer</div>
            
            <div className="flex-1 overflow-y-auto px-2 space-y-1">
                <div 
                    onClick={() => setActiveTab('terminal')}
                    className={`flex items-center space-x-2 px-3 py-2 rounded cursor-pointer transition-all duration-200 ${activeTab === 'terminal' ? 'bg-[#1a1a1a] text-green-400 border-l-2 border-green-500' : 'hover:bg-[#111] hover:text-gray-100'}`}
                >
                    <Terminal size={16} />
                    <span className="text-sm font-medium">Terminal.sh</span>
                </div>

                <div className="mt-4 mb-2 px-3 flex items-center space-x-2 text-gray-500 text-xs">
                    <ChevronRight size={12} className="rotate-90"/>
                    <span>SRC / BACKEND</span>
                </div>

                {projects.map(p => (
                    <div 
                        key={p.id}
                        onClick={() => setActiveTab(p.id)}
                        className={`flex items-center space-x-2 px-3 py-2 rounded cursor-pointer transition-all duration-200 ${activeTab === p.id ? 'bg-[#1a1a1a] text-blue-400 border-l-2 border-blue-500' : 'hover:bg-[#111] hover:text-gray-100'}`}
                    >
                        {p.icon}
                        <span className="text-sm">{p.name}</span>
                    </div>
                ))}

                <div className="mt-6 border-t border-[#222] pt-4 space-y-2">
                    <a href="#" className="flex items-center space-x-2 px-3 py-2 text-gray-500 hover:text-white transition-colors">
                        <Github size={16} />
                        <span className="text-xs">GitHub Repo</span>
                    </a>
                    <a href="#" className="flex items-center space-x-2 px-3 py-2 text-gray-500 hover:text-white transition-colors">
                        <Linkedin size={16} />
                        <span className="text-xs">LinkedIn Profile</span>
                    </a>
                </div>
            </div>
        </div>

        {/* EDITOR / TERMINAL AREA */}
        <div className="flex-1 flex flex-col bg-[#0a0a0a] relative">
            
            {/* Tabs */}
            <div className="flex bg-[#0d0d0d] border-b border-[#222] overflow-x-auto no-scrollbar">
                <div 
                    onClick={() => setActiveTab('terminal')}
                    className={`flex items-center space-x-2 px-6 py-2 text-xs cursor-pointer border-r border-[#222] ${activeTab === 'terminal' ? 'bg-[#0a0a0a] text-green-400 border-t-2 border-t-green-500' : 'text-gray-500 hover:bg-[#111]'}`}
                >
                    <Terminal size={12} />
                    <span>Terminal</span>
                </div>
                {projects.map(p => (
                    <div 
                        key={p.id}
                        onClick={() => setActiveTab(p.id)}
                        className={`flex items-center space-x-2 px-6 py-2 text-xs cursor-pointer border-r border-[#222] ${activeTab === p.id ? 'bg-[#0a0a0a] text-blue-400 border-t-2 border-t-blue-500' : 'text-gray-500 hover:bg-[#111]'}`}
                    >
                        <Code size={12} />
                        <span>{p.name}</span>
                    </div>
                ))}
            </div>

            {/* Content */}
            <div className="flex-1 p-6 overflow-y-auto" onClick={focusInput}>
                
                {/* TERMINAL VIEW */}
                {activeTab === 'terminal' && (
                    <div className="max-w-3xl mx-auto font-mono text-sm space-y-2 pb-20">
                        {history.map((line, i) => (
                            <div key={i} className="animate-fade-in break-words flex">
                                {line.type === 'user' && <span className="text-green-500 font-bold mr-2">➜  ~</span>}
                                {line.type === 'system' && <span className="text-blue-500 mr-2">[SYS]</span>}
                                {line.type === 'success' && <span className="text-green-400 mr-2">✔</span>}
                                {line.type === 'error' && <span className="text-red-500 mr-2">✖</span>}
                                {line.type === 'info' && <span className="text-yellow-500 mr-2">ℹ</span>}
                                
                                <div className={`flex-1 ${line.type === 'user' ? 'text-gray-100' : 'text-gray-400'}`}>
                                    {line.type === 'list' ? (
                                             <div className="grid grid-cols-2 gap-2 mt-1">
                                                {line.content.map((cmd, idx) => (
                                                    <button
                                                        key={idx}
                                                        type="button"
                                                        onClick={() => { setInput(cmd); inputRef.current?.focus(); }}
                                                        className="bg-[#111] p-2 rounded border border-[#222] text-green-400 text-xs hover:border-green-900 transition-colors text-left"
                                                    >
                                                        {cmd}
                                                    </button>
                                                ))}
                                             </div>
                                        ) : (
                                            <span className={line.type === 'user' ? 'text-white' : ''}>{line.content}</span>
                                        )}
                                </div>
                            </div>
                        ))}
                        
                        {/* Input Area */}
                        <div className="flex items-center mt-4 group">
                            <span className="text-green-500 font-bold mr-2 text-glow">➜  ~</span>
                            <input 
                                ref={inputRef}
                                type="text" 
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                className="bg-transparent outline-none text-gray-100 w-full font-mono placeholder-gray-700 caret-green-500"
                                placeholder="Type 'help' to start..."
                                autoFocus
                            />
                        </div>
                        <div ref={bottomRef} />
                    </div>
                )}

                {/* CODE VIEW */}
                {projects.map(p => activeTab === p.id && (
                    <div key={p.id} className="max-w-4xl mx-auto animate-fade-in">
                        <div className="mb-6 flex items-start justify-between">
                            <div>
                                <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                                    {p.name}
                                    <span className="text-xs font-normal px-2 py-0.5 rounded-full border border-gray-700 text-gray-500">READ ONLY</span>
                                </h1>
                                <p className="text-gray-400 text-sm">{p.desc}</p>
                            </div>
                            <button 
                                type="button"
                                onClick={() => {
                                    setActiveTab('terminal');
                                    processCommand(`run ${p.lang}`);
                                }}
                                className="bg-green-900/30 text-green-400 hover:bg-green-900/50 hover:text-green-200 px-4 py-2 rounded text-xs font-bold border border-green-900 transition-all flex items-center gap-2"
                            >
                                <Play size={14} /> RUN BUILD
                            </button>
                        </div>

                        <div className="bg-[#0d0d0d] border border-[#222] rounded-lg p-4 relative overflow-hidden group">
                            <div className="absolute top-0 right-0 p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                <span className="text-xs text-gray-600">UTF-8</span>
                            </div>
                            <pre className="font-mono text-sm text-blue-300 leading-relaxed overflow-x-auto">
                                <code>{p.code}</code>
                            </pre>
                        </div>
                    </div>
                ))}

            </div>

            {/* STATUS BAR */}
            <div className="h-6 bg-[#007acc] flex items-center justify-between px-3 text-[10px] text-white select-none z-20">
                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                        <Wifi size={10} />
                        <span>REMOTE: NYC_SERVER_01</span>
                    </div>
                    <div className="flex items-center space-x-1">
                        <User size={10} />
                        <span>GUEST SESSION</span>
                    </div>
                </div>
                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                        <span>Ln 1, Col 1</span>
                    </div>
                    <div className="flex items-center space-x-1">
                        <Globe size={10} />
                        <span>UTF-8</span>
                    </div>
                    <span>{time}</span>
                </div>
            </div>

        </div>
      </div>
    </div>
  );
};

export default App;