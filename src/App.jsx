import {
  Terminal, Code, User, Briefcase, Mail, Cpu, X, Minus, Square,
  Play, Folder, FileText, ChevronRight, Github, Linkedin,
  Globe, Database, Server, Wifi, Zap, Layers, Lock, Box, BookOpen, Award, Users
} from 'lucide-react';

import React, { useState, useEffect, useRef } from 'react';

const App = () => {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([
    { type: 'system', content: '> INITIALIZING BHAVESH_KERNEL v4.0...' },
    { type: 'system', content: '> LOADING MODULES: [DISTRIBUTED_SYSTEMS, ML_PIPELINES, CLOUD_NATIVE]...' },
    { type: 'success', content: '> SYSTEM READY. WELCOME, GUEST.' },
    { type: 'info', content: 'Type "help" to view available commands.' },
  ]);
  const [activeTab, setActiveTab] = useState('terminal');
  const [isBooting, setIsBooting] = useState(true);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const commandHistoryRef = useRef([]);
  const [historyIndex, setHistoryIndex] = useState(null);
  const portfolioRef = useRef(null);

  // Boot Sequence
  useEffect(() => {
    setTimeout(() => setIsBooting(false), 2800);
  }, []);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  const focusInput = () => {
    if (activeTab === 'terminal') inputRef.current?.focus();
  }

  const commands = {
    help: {
      text: "Available commands:",
      list: ["about", "experience", "skills", "projects", "publications", "leadership", "contact", "clear", "gui"]
    },
    about: "Bhavesh Gupta. MSCS @ NYU Courant ('27). Ex-System Engineer @ TCS. Passionate about Distributed Systems, HPC, and AI/ML.",
    experience: {
      text: "Work History:",
      list: [
        "2024-25: System Engineer @ TCS (Java/SQL, Financial Trading System, 26M+ txns/week)",
        "2023: ML/AI Intern @ Synopsys (Deep Learning, Drug Discovery, 98% Accuracy)",
        "2022: Android Dev Intern @ eSec Forte (Flutter, Security, Penetration Testing)"
      ]
    },
    skills: "LANGUAGES: Python, Java, C++, SQL, Kotlin | TECH: AWS, Docker, Kubernetes, PyTorch, Spring Boot, FastAPI",
    publications: "Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification (ICAIA 2024)",
    leadership: {
      text: "Leadership & Extra-Curriculars:",
      list: [
        "Android Lead @ Google Developer Student Club (Taught 50+ students)",
        "President @ University Science Club (Organized TechFest for 10k+ students)",
        "Founding Member Mentor @ USICT Alumni Committee"
      ]
    },
    contact: "Email: bg2896@nyu.edu | LinkedIn: /in/bhaveshgupta01 | GitHub: @bhaveshgupta01",
    clear: "CLEAR_ACTION",
    gui: "SCROLL_ACTION"
  };

  const projects = [
    {
      id: 'split-dnn',
      name: 'Split-DNN.py',
      lang: 'python',
      icon: <Layers size={16} className="text-blue-400" />,
      desc: 'Privacy-preserving edge AI framework.',
      code: `class SplitDNN(nn.Module):
    def __init__(self, client_model, server_model):
        super().__init__()
        self.client = client_model
        self.server = server_model
        self.quantizer = INT8Quantizer()

    def forward(self, x):
        # Edge Computation
        intermediate = self.client(x)
        # Privacy Preservation
        quantized = self.quantizer(intermediate)
        # Server Computation
        return self.server(quantized)`
    },
    {
      id: 'trading',
      name: 'TradeEngine.java',
      lang: 'java',
      icon: <Database size={16} className="text-orange-400" />,
      desc: 'High-frequency trading system.',
      code: `public class MatchingEngine {
    private final OrderBook book;
    
    public void process(Order order) {
        // 26M+ weekly transactions
        // Sub-millisecond latency
        if (book.match(order)) {
            eventBus.publish(new TradeEvent(order));
        }
    }
}`
    },
    {
      id: 'bunkey',
      name: 'BunKey.kt',
      lang: 'kotlin',
      icon: <Lock size={16} className="text-purple-400" />,
      desc: 'E2E Encryption Android App.',
      code: `fun encrypt(data: ByteArray, key: SecretKey): ByteArray {
    val cipher = Cipher.getInstance("AES/GCM/NoPadding")
    cipher.init(Cipher.ENCRYPT_MODE, key)
    return cipher.doFinal(data)
    // Custom dual-key architecture
}`
    },
    {
      id: 'portfolio',
      name: 'Portfolio.docker',
      lang: 'docker',
      icon: <Box size={16} className="text-cyan-400" />,
      desc: 'Containerized Terminal Portfolio.',
      code: `FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# Live system metrics stream
EXPOSE 8080
CMD ["npm", "run", "dev"]`
    }
  ];

  const processCommand = (cmdRaw) => {
    const trimmed = (cmdRaw || '').trim();
    if (!trimmed) return;

    const args = trimmed.toLowerCase().split(' ');
    const mainCmd = args[0];

    setHistory(prev => [...prev, { type: 'user', content: trimmed }]);
    commandHistoryRef.current.push(trimmed);
    setHistoryIndex(commandHistoryRef.current.length);
    setInput('');

    if (mainCmd === 'clear') {
      setHistory([]);
      return;
    }

    if (mainCmd === 'gui' || mainCmd === 'scroll') {
      portfolioRef.current?.scrollIntoView({ behavior: 'smooth' });
      setHistory(prev => [...prev, { type: 'success', content: 'Opening GUI profile...' }]);
      return;
    }

    if (commands[mainCmd]) {
      const res = commands[mainCmd];
      if (typeof res === 'object' && res.list) {
        setHistory(prev => [...prev, { type: 'info', content: res.text }, { type: 'list', content: res.list }]);
      } else {
        setHistory(prev => [...prev, { type: 'success', content: res }]);
      }
      return;
    }

    if (mainCmd === 'run') {
      const lang = args[1];
      const proj = projects.find(p => p.lang.includes(lang) || p.name.toLowerCase().includes(lang));

      if (proj) {
        setHistory(prev => [...prev, { type: 'info', content: `Compiling ${proj.name}...` }]);
        setTimeout(() => {
          setHistory(prev => [...prev,
          { type: 'success', content: `Build Successful (12ms)` },
          { type: 'output', content: `> Executing artifact...` },
          { type: 'code', content: proj.code },
          { type: 'output', content: `> Process finished with exit code 0` },
          ]);
        }, 600);
      } else {
        setHistory(prev => [...prev, { type: 'error', content: `Error: Module '${lang}' not found. Available: python, java, kotlin, docker` }]);
      }
      return;
    }

    setHistory(prev => [...prev, { type: 'error', content: `Command not found: ${mainCmd}. Type 'help'.` }]);
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
      setInput(commandHistoryRef.current[next] || '');
      setHistoryIndex(next);
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const idx = historyIndex === null ? commandHistoryRef.current.length : historyIndex;
      const next = Math.min(commandHistoryRef.current.length, (idx || 0) + 1);
      setInput(commandHistoryRef.current[next] || '');
      setHistoryIndex(next === commandHistoryRef.current.length ? null : next);
    }
  };

  if (isBooting) {
    return (
      <div className="h-screen w-full bg-[#050505] flex flex-col items-center justify-center font-mono text-slate-300 p-4">
        <div className="w-full max-w-md space-y-4 flex flex-col items-center">
          <div className="w-16 h-16 mb-4 relative">
            <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-20 animate-pulse"></div>
            <img src="/bg-logo.png" alt="Logo" className="w-full h-full object-contain relative z-10 opacity-90" style={{ filter: 'drop-shadow(0 0 8px rgba(6,182,212,0.5))' }} />
          </div>
          <div className="flex justify-between w-full text-xs text-slate-500 uppercase tracking-widest">
            <span>System Boot</span>
            <span>Ver 4.0.1</span>
          </div>
          <div className="h-0.5 w-full bg-slate-800 rounded overflow-hidden">
            <div className="h-full bg-cyan-500 animate-progress w-full origin-left"></div>
          </div>
          <div className="text-xs space-y-1 text-slate-400 w-full font-mono">
            <p className="animate-fade-in" style={{ animationDelay: '0.2s' }}>{'>'} MOUNTING VOLUMES...</p>
            <p className="animate-fade-in" style={{ animationDelay: '0.8s' }}>{'>'} LOADING ASSETS...</p>
            <p className="animate-fade-in" style={{ animationDelay: '1.5s' }}>{'>'} ESTABLISHING SECURE CONNECTION...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-[#050505] text-slate-300 font-mono relative selection:bg-cyan-900 selection:text-cyan-100 h-screen overflow-y-scroll snap-y snap-mandatory">

      {/* Terminal Section */}
      <section className="relative h-screen w-full flex flex-col snap-start">
        {/* Subtle Grid Background */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px] pointer-events-none"></div>

        {/* Top Bar */}
        <div className="h-12 bg-[#0a0a0a] border-b border-white/5 flex items-center justify-between px-4 z-10 select-none backdrop-blur-sm bg-opacity-80">
          <div className="flex items-center space-x-3">
            <div className="flex space-x-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50"></div>
            </div>
            <span className="ml-4 text-xs text-slate-500 font-medium tracking-wide flex items-center gap-2">
              <Cpu size={12} />
              bhavesh@nyu-courant:~
            </span>
          </div>
          <div className="flex items-center space-x-4 text-slate-600">
            <span className="text-xs flex items-center gap-1"><Wifi size={12} /> CONNECTED</span>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden relative z-10">
          {/* Sidebar */}
          <div className="w-64 bg-[#080808] border-r border-white/5 flex flex-col hidden md:flex">
            <div className="p-4 text-xs font-bold text-slate-500 tracking-widest uppercase flex items-center gap-2">
              <Folder size={14} /> Project Explorer
            </div>
            <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
              <div className="flex items-center space-x-2 px-3 py-2 rounded bg-white/5 text-cyan-400 border-l-2 border-cyan-500">
                <Terminal size={14} />
                <span className="text-sm">terminal.zsh</span>
              </div>

              <div className="mt-6 mb-2 px-3 flex items-center space-x-2 text-slate-600 text-xs font-bold">
                <ChevronRight size={10} className="rotate-90" />
                <span>PROJECTS</span>
              </div>
              {projects.map(p => (
                <div key={p.id} className="group flex items-center space-x-2 px-3 py-2 rounded cursor-pointer transition-all duration-200 hover:bg-white/5 text-slate-400 hover:text-slate-200">
                  <div className="opacity-70 group-hover:opacity-100 transition-opacity">{p.icon}</div>
                  <span className="text-sm">{p.name}</span>
                </div>
              ))}
            </div>

            <div className="p-4 border-t border-white/5 space-y-2">
              <div className="text-xs text-slate-600 mb-2 font-bold">CONTACT</div>
              <a href="https://github.com/bhaveshgupta01" target="_blank" className="flex items-center gap-2 text-slate-400 hover:text-white text-xs transition-colors">
                <Github size={14} /> @bhaveshgupta01
              </a>
              <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" className="flex items-center gap-2 text-slate-400 hover:text-white text-xs transition-colors">
                <Linkedin size={14} /> /in/bhaveshgupta01
              </a>
            </div>
          </div>

          {/* Terminal */}
          <div className="flex-1 flex flex-col bg-[#050505]/95 relative" onClick={focusInput}>
            <div className="flex-1 overflow-y-auto p-6 font-mono text-sm space-y-2 pb-20 scrollbar-hide">
              {history.map((line, i) => (
                <div key={i} className="animate-fade-in break-words">
                  <div className="flex">
                    {line.type === 'user' && <span className="text-cyan-500 font-bold mr-3">➜ ~</span>}
                    {line.type === 'system' && <span className="text-slate-500 mr-3">#</span>}
                    {line.type === 'success' && <span className="text-green-500 mr-3">✔</span>}
                    {line.type === 'error' && <span className="text-red-500 mr-3">✖</span>}
                    {line.type === 'info' && <span className="text-blue-400 mr-3">ℹ</span>}
                    {line.type === 'output' && <span className="text-slate-500 mr-3"></span>}

                    <div className={`flex-1 ${line.type === 'user' ? 'text-slate-100' : 'text-slate-400'}`}>
                      {line.type === 'list' ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                          {line.content.map((cmd, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-slate-300">
                              <span className="text-cyan-500">○</span> {cmd}
                            </div>
                          ))}
                        </div>
                      ) : line.type === 'code' ? (
                        <div className="mt-2 p-3 bg-white/5 rounded border border-white/10 text-xs font-mono text-slate-300 whitespace-pre overflow-x-auto">
                          {line.content}
                        </div>
                      ) : (
                        <span>{line.content}</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              <div className="flex items-center mt-4">
                <span className="text-cyan-500 font-bold mr-3">➜ ~</span>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="bg-transparent outline-none text-slate-100 w-full font-mono placeholder-slate-700"
                  placeholder="Type 'help'..."
                  autoFocus
                />
              </div>
              <div ref={bottomRef} />
            </div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-slate-600 animate-bounce cursor-pointer z-20" onClick={() => portfolioRef.current?.scrollIntoView({ behavior: 'smooth' })}>
          <span className="text-[10px] uppercase tracking-widest">Scroll for GUI</span>
          <ChevronRight size={16} className="rotate-90" />
        </div>
      </section>

      {/* Portfolio Sections Container */}
      <div ref={portfolioRef}>
        <AboutSection />
        <ExperienceSection />
        <SkillsSection />
        <ProjectsSection projects={projects} />
        <LeadershipSection />
        <ContactSection />
      </div>
    </div>
  );
};

// --- SECTIONS ---

const AboutSection = () => (
  <section className="min-h-screen w-full bg-[#050505] flex items-center justify-center p-4 md:p-20 relative snap-start overflow-hidden">
    {/* Background Gradients */}
    <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
      <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-cyan-900/20 rounded-full blur-[120px]"></div>
      <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] bg-purple-900/20 rounded-full blur-[120px]"></div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] opacity-10 pointer-events-none z-0">
        <img src="/bg-logo.png" alt="" className="w-full h-full object-contain" />
      </div>
    </div>

    <div className="max-w-6xl w-full grid grid-cols-1 md:grid-cols-12 gap-12 relative z-10 items-center">
      <div className="md:col-span-7 space-y-8">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-900/30 text-cyan-400 border border-cyan-800/50 text-xs font-medium">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            Open to Opportunities
          </div>
          <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight">
            Bhavesh <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">Gupta</span>
          </h1>
          <p className="text-xl text-slate-400 font-light">
            MSCS @ NYU Courant '27
          </p>
        </div>

        <p className="text-slate-400 leading-relaxed text-lg">
          Software Engineer with a passion for <span className="text-slate-200">Distributed Systems</span>, <span className="text-slate-200">High-Performance Computing</span>, and <span className="text-slate-200">AI/ML</span>.
          Previously architected financial trading systems at TCS and optimized deep learning models at Synopsys.
        </p>
      </div>

      <div className="md:col-span-5 flex justify-center md:justify-end">
        {/* Flip Card */}
        <div className="relative w-64 h-64 md:w-80 md:h-80 group perspective-1000">
          <div className="w-full h-full relative transform-style-3d transition-transform duration-700 group-hover:rotate-y-180">
            {/* Front Face (Panda) */}
            <div className="absolute inset-0 backface-hidden">
              <div className="absolute inset-0 bg-gradient-to-tr from-cyan-500 to-purple-500 rounded-2xl blur-2xl opacity-20"></div>
              <div className="absolute inset-0 bg-[#0a0a0a] rounded-2xl border border-white/10 flex items-center justify-center overflow-hidden">
                <img src="/panda.png" alt="Panda" className="w-full h-full object-cover" />
              </div>
            </div>
            {/* Back Face (Profile Photo) */}
            <div className="absolute inset-0 backface-hidden rotate-y-180">
              <div className="absolute inset-0 bg-gradient-to-tr from-green-500 to-blue-500 rounded-2xl blur-2xl opacity-20"></div>
              <div className="absolute inset-0 bg-[#0a0a0a] rounded-2xl border border-white/10 flex items-center justify-center overflow-hidden">
                <img src="/profile_photo.png" alt="Profile" className="w-full h-full object-cover" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const ExperienceSection = () => (
  <section className="min-h-screen w-full bg-[#050505] flex items-center justify-center p-4 md:p-20 relative snap-start">
    <div className="max-w-4xl w-full">
      <h2 className="text-4xl font-bold text-white mb-12 flex items-center gap-4">
        <Briefcase className="text-cyan-500" /> Experience
      </h2>

      <div className="space-y-8 border-l-2 border-white/10 pl-8 ml-4">
        <div className="relative group">
          <div className="absolute -left-[41px] top-1.5 w-5 h-5 rounded-full bg-[#050505] border-4 border-cyan-500 group-hover:scale-125 transition-transform"></div>
          <div className="bg-white/5 p-6 rounded-xl border border-white/10 hover:border-cyan-500/30 transition-all">
            <h3 className="text-xl font-bold text-white">System Engineer @ Tata Consultancy Services</h3>
            <p className="text-cyan-400 text-sm mb-2">June 2024 - July 2025</p>
            <p className="text-slate-400">
              Architected a high-frequency Java/SQL trading system handling 26M+ transactions/week.
              Led the development of GenAI validation tools to automate testing workflows.
            </p>
          </div>
        </div>

        <div className="relative group">
          <div className="absolute -left-[41px] top-1.5 w-5 h-5 rounded-full bg-[#050505] border-4 border-purple-500 group-hover:scale-125 transition-transform"></div>
          <div className="bg-white/5 p-6 rounded-xl border border-white/10 hover:border-purple-500/30 transition-all">
            <h3 className="text-xl font-bold text-white">ML/AI Intern @ Synopsys Inc.</h3>
            <p className="text-purple-400 text-sm mb-2">July 2023 - Sept 2023</p>
            <p className="text-slate-400">
              Engineered Deep Learning models for drug-target binding prediction achieving 98% accuracy.
              Optimized model inference time by 40% using quantization techniques.
            </p>
          </div>
        </div>

        <div className="relative group">
          <div className="absolute -left-[41px] top-1.5 w-5 h-5 rounded-full bg-[#050505] border-4 border-green-500 group-hover:scale-125 transition-transform"></div>
          <div className="bg-white/5 p-6 rounded-xl border border-white/10 hover:border-green-500/30 transition-all">
            <h3 className="text-xl font-bold text-white">Android Dev Intern @ eSec Forte</h3>
            <p className="text-green-400 text-sm mb-2">May 2022 - July 2022</p>
            <p className="text-slate-400">
              Developed secure Android applications using Flutter. Implemented penetration testing protocols
              to ensure app security and data privacy.
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>
);

const SkillsSection = () => (
  <section className="min-h-screen w-full bg-[#050505] flex items-center justify-center p-4 md:p-20 relative snap-start">
    <div className="max-w-4xl w-full">
      <h2 className="text-4xl font-bold text-white mb-12 flex items-center gap-4">
        <Cpu className="text-purple-500" /> Technical Arsenal
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-300">Languages</h3>
          <div className="flex flex-wrap gap-2">
            {['Python', 'Java', 'C++', 'SQL', 'Kotlin', 'JavaScript'].map(skill => (
              <span key={skill} className="px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-slate-300 text-sm hover:border-purple-500/50 hover:text-purple-400 transition-colors cursor-default">
                {skill}
              </span>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-300">Technologies</h3>
          <div className="flex flex-wrap gap-2">
            {['AWS', 'Docker', 'Kubernetes', 'PyTorch', 'Spring Boot', 'FastAPI', 'Git', 'Linux'].map(skill => (
              <span key={skill} className="px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-slate-300 text-sm hover:border-cyan-500/50 hover:text-cyan-400 transition-colors cursor-default">
                {skill}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  </section>
);

const ProjectsSection = ({ projects }) => (
  <section className="min-h-screen w-full bg-[#050505] flex items-center justify-center p-4 md:p-20 relative snap-start">
    <div className="max-w-6xl w-full">
      <h2 className="text-4xl font-bold text-white mb-12 flex items-center gap-4">
        <Folder className="text-yellow-500" /> Featured Projects
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {projects.map((project) => (
          <div key={project.id} className="group bg-white/5 p-6 rounded-xl border border-white/10 hover:border-yellow-500/30 transition-all hover:-translate-y-1">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 rounded-lg bg-white/5 text-yellow-500">
                {project.icon}
              </div>
              <span className="text-xs font-mono text-slate-500">{project.lang}</span>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">{project.name}</h3>
            <p className="text-slate-400 text-sm mb-4">{project.desc}</p>
            <div className="bg-black/50 p-3 rounded border border-white/5 font-mono text-xs text-slate-500 overflow-hidden h-20 relative">
              <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/90"></div>
              <pre>{project.code}</pre>
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

const LeadershipSection = () => (
  <section className="min-h-screen w-full bg-[#050505] flex items-center justify-center p-4 md:p-20 relative snap-start">
    <div className="max-w-4xl w-full space-y-16">

      {/* Publications */}
      <div>
        <h2 className="text-4xl font-bold text-white mb-8 flex items-center gap-4">
          <BookOpen className="text-pink-500" /> Publications
        </h2>
        <div className="bg-white/5 p-6 rounded-xl border border-white/10 hover:border-pink-500/30 transition-all">
          <p className="text-lg text-white font-medium italic">"Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification"</p>
          <p className="text-slate-400 mt-2">Proceedings on Engineering Sciences (PES), Conference: ICAIA 2024</p>
        </div>
      </div>

      {/* Leadership */}
      <div>
        <h2 className="text-4xl font-bold text-white mb-8 flex items-center gap-4">
          <Award className="text-orange-500" /> Leadership
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white/5 p-6 rounded-xl border border-white/10 hover:border-orange-500/30 transition-all">
            <h3 className="text-lg font-bold text-white mb-2">Android Lead @ GDSC</h3>
            <p className="text-slate-400 text-sm">Mentored 50+ students in Android Development. Organized workshops and hackathons.</p>
          </div>
          <div className="bg-white/5 p-6 rounded-xl border border-white/10 hover:border-orange-500/30 transition-all">
            <h3 className="text-lg font-bold text-white mb-2">President @ Science Club</h3>
            <p className="text-slate-400 text-sm">Led a team of 20 to organize the University TechFest with 10k+ attendees.</p>
          </div>
        </div>
      </div>

    </div>
  </section>
);

const ContactSection = () => (
  <section className="min-h-screen w-full bg-[#050505] flex items-center justify-center p-4 md:p-20 relative snap-start">
    <div className="max-w-4xl w-full text-center">
      <h2 className="text-5xl font-bold text-white mb-8">Let's Connect</h2>
      <p className="text-xl text-slate-400 mb-12">
        Always open to discussing new opportunities, distributed systems, or just chatting about tech.
      </p>

      <div className="flex flex-wrap justify-center gap-6">
        <a href="https://github.com/bhaveshgupta01" target="_blank" className="flex items-center gap-3 px-8 py-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white font-bold text-lg transition-all hover:-translate-y-1">
          <Github size={24} />
          <span>GitHub</span>
        </a>
        <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" className="flex items-center gap-3 px-8 py-4 rounded-xl bg-[#0077b5]/10 hover:bg-[#0077b5]/20 border border-[#0077b5]/30 text-[#0077b5] font-bold text-lg transition-all hover:-translate-y-1">
          <Linkedin size={24} />
          <span>LinkedIn</span>
        </a>
        <a href="mailto:bg2896@nyu.edu" className="flex items-center gap-3 px-8 py-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 font-bold text-lg transition-all hover:-translate-y-1">
          <Mail size={24} />
          <span>Email Me</span>
        </a>
      </div>

      <footer className="mt-20 text-slate-600 text-sm">
        <p>© 2025 Bhavesh Gupta. Built with React + Vite.</p>
      </footer>
    </div>
  </section>
);

export default App;