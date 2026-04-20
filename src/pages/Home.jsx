import React, { useState, useEffect, useRef } from 'react';
import {
  User, Briefcase, Cpu, Folder, Palette, Mail,
  Github, Linkedin, ChevronDown, Award, BookOpen, ExternalLink,
  X, Map, Menu,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ChatIsland from '../components/ChatIsland';
import ThemeToggle from '../components/ThemeToggle';
import BackgroundAmbient from '../components/BackgroundAmbient';
import BrandMark from '../components/BrandMark';
import ScrollDrivenLogo from '../components/ScrollDrivenLogo';
import LogoIcon from '../components/LogoIcon';
import { Download, Phone, MapPin, QrCode } from 'lucide-react';
import { downloadVCard } from '../lib/vcard';

const API_URL = import.meta.env.VITE_AGENT_API || 'http://localhost:8000';

const SECTION_IDS = [
  'about', 'experience', 'skills', 'projects',
  'publications', 'leadership', 'contact', 'journey',
];

// Sidebar nav — order MATCHES on-page scroll order.
// Page: Hero → About → Journey → Experience → Skills → Projects → Leadership → Publications → Contact
const NAV_ITEMS = [
  { id: 'about',        label: 'About Me',     icon: User },
  { id: 'journey',      label: 'Journey',      icon: Map },
  { id: 'experience',   label: 'Experience',   icon: Briefcase },
  { id: 'skills',       label: 'Tech Stack',   icon: Cpu },
  { id: 'projects',     label: 'Projects',     icon: Folder },
  { id: 'leadership',   label: 'Leadership',   icon: Award },
  { id: 'publications', label: 'Publications', icon: BookOpen },
  { id: 'design',       label: 'Designs',      icon: Palette,  external: true },
  { id: 'contact',      label: 'Contact',      icon: Mail },
];

const Home = () => {
  const navigate = useNavigate();

  const [isTyping, setIsTyping] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);

  // --- ui state ---
  const [isScrolling, setIsScrolling] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const scrollTimerRef = useRef(null);

  const refs = {
    hero: useRef(null),
    about: useRef(null),
    journey: useRef(null),
    experience: useRef(null),
    skills: useRef(null),
    projects: useRef(null),
    leadership: useRef(null),
    publications: useRef(null),
    contact: useRef(null),
  };
  const chatBottomRef = useRef(null);
  const scrollContainerRef = useRef(null);

  // Accent palette tokens — all map back to theme-aware CSS variables.
  const accents = {
    rose:    'text-rose-500 bg-rose-500/10 border-rose-500/20',
    amber:   'text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20',
    emerald: 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    blue:    'text-blue-600 dark:text-blue-400 bg-blue-500/10 border-blue-500/20',
    purple:  'text-violet-600 dark:text-violet-400 bg-violet-500/10 border-violet-500/20',
  };

  const suggestionChips = [
    { label: 'About Me',     cmd: 'Who are you?',                            icon: <User size={14}/> },
    { label: 'Experience',   cmd: 'Walk me through your work experience.',   icon: <Briefcase size={14}/> },
    { label: 'Projects',     cmd: 'What are your favorite projects?',        icon: <Folder size={14}/> },
    { label: 'Tech Stack',   cmd: 'What is your tech stack?',                icon: <Cpu size={14}/> },
    { label: 'Designs',      cmd: 'design',                                  icon: <Palette size={14}/> },
    { label: 'Publications', cmd: 'Tell me about your published research.',  icon: <BookOpen size={14}/> },
    { label: 'Contact',      cmd: 'How do I contact you?',                   icon: <Mail size={14}/> },
    { label: 'Clear',        cmd: 'clear',                                   icon: <X size={14}/> },
  ];

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Scroll listener → triggers adaptive-opacity on the chat island.
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const onScroll = () => {
      setIsScrolling(true);
      if (scrollTimerRef.current) clearTimeout(scrollTimerRef.current);
      scrollTimerRef.current = setTimeout(() => setIsScrolling(false), 650);
    };
    container.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      container.removeEventListener('scroll', onScroll);
      if (scrollTimerRef.current) clearTimeout(scrollTimerRef.current);
    };
  }, []);

  const scrollTo = (id) => {
    if (id && SECTION_IDS.includes(id) && refs[id]?.current) {
      refs[id].current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleNav = (item) => {
    setMobileMenuOpen(false);
    if (item.external && item.id === 'design') {
      navigate('/design');
    } else {
      scrollTo(item.id);
    }
  };

  const handleSend = async (text = input) => {
    const q = (text || '').trim();
    if (!q) return;

    if (q.toLowerCase() === 'clear') { setMessages([]); setInput(''); return; }
    if (q.toLowerCase() === 'design') { navigate('/design'); setInput(''); return; }

    const userMsg = { id: Date.now(), role: 'user', text: q };
    setMessages(prev => [...prev, userMsg]);
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
      if (data.scroll_to) scrollTo(data.scroll_to);

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: data.answer || '…',
        component: data.component || null,
        source: data.source || 'profile',
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: "My agent is offline right now — try the sidebar, or email me at bg2896@nyu.edu.",
        error: true,
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex h-screen w-full text-ink font-sans overflow-hidden relative selection:bg-accent/25">
      <BackgroundAmbient />

      {/* ===== DESKTOP SIDEBAR ===== */}
      <aside className="w-64 flex-shrink-0 bg-surface border-r border-subtle/15 flex-col z-20 h-full hidden md:flex">
        <div className="p-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BrandMark size={36} />
            <span className="font-semibold text-[15px] tracking-tight text-ink">
              Bhavesh<span className="text-accent">.ai</span>
            </span>
          </div>
          <ThemeToggle />
        </div>

        <nav className="flex-1 px-3 mt-2 overflow-y-auto">
          {NAV_ITEMS.map(item => (
            <SidebarItem
              key={item.id}
              icon={item.icon}
              label={item.label}
              onClick={() => handleNav(item)}
            />
          ))}
        </nav>

        <div className="p-4 border-t border-subtle/15">
          <div className="flex items-center gap-3 mb-3">
            <a href="https://github.com/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="text-faint hover:text-ink transition-colors">
              <Github size={15}/>
            </a>
            <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="text-faint hover:text-ink transition-colors">
              <Linkedin size={15}/>
            </a>
            <a href="mailto:bg2896@nyu.edu" className="text-faint hover:text-ink transition-colors">
              <Mail size={15}/>
            </a>
          </div>
          <div className="text-[10px] text-faint leading-relaxed">
            first-person agent grounded in my resume
          </div>
        </div>
      </aside>

      {/* ===== MOBILE TOP BAR ===== */}
      <div className="md:hidden fixed top-0 inset-x-0 z-40 glass-strong flex items-center justify-between px-4 h-14">
        <div className="flex items-center gap-2.5 relative z-10">
          <BrandMark size={30} />
          <span className="font-semibold text-[14px] tracking-tight text-ink">
            Bhavesh<span className="text-accent">.ai</span>
          </span>
        </div>
        <div className="flex items-center gap-2 relative z-10">
          <ThemeToggle />
          <button
            onClick={() => setMobileMenuOpen(v => !v)}
            className="glass-chip w-9 h-9 rounded-full flex items-center justify-center text-muted"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X size={15}/> : <Menu size={15}/>}
          </button>
        </div>
      </div>

      {/* ===== MOBILE MENU OVERLAY ===== */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            className="md:hidden fixed inset-0 top-14 z-30 bg-app/90 backdrop-blur-2xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={() => setMobileMenuOpen(false)}
          >
            <motion.div
              className="flex flex-col px-6 py-8 gap-1"
              initial={{ y: -12 }}
              animate={{ y: 0 }}
              exit={{ y: -12 }}
              transition={{ duration: 0.25 }}
              onClick={(e) => e.stopPropagation()}
            >
              {NAV_ITEMS.map(item => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNav(item)}
                    className="flex items-center gap-3 px-4 py-3.5 rounded-xl text-muted hover:text-ink hover:bg-surface-2 transition-all text-left"
                  >
                    <Icon size={18}/>
                    <span className="font-medium text-[15px]">{item.label}</span>
                  </button>
                );
              })}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* scroll-driven logo watermark — behind all content, tracks hero exit */}
      <ScrollDrivenLogo scrollContainerRef={scrollContainerRef} />

      {/* ===== MAIN SCROLL ===== */}
      <main className="flex-1 relative h-full w-full">
        <div ref={scrollContainerRef} className="absolute inset-0 overflow-y-auto scroll-smooth snap-y snap-mandatory pb-48 z-0" id="main-scroll">

          {/* HERO */}
          <section ref={refs.hero} className="h-screen w-full snap-start flex flex-col items-center justify-center p-6 text-center relative">
             <button
               type="button"
               onClick={downloadVCard}
               className="mb-8 relative group cursor-pointer z-10"
               aria-label="Download contact card"
             >
                <div className="absolute inset-0 bg-accent blur-[60px] opacity-20 rounded-full group-hover:opacity-40 transition-opacity"></div>
                <div className="w-40 h-40 bg-white p-2 rounded-2xl shadow-2xl rotate-3 transition-transform group-hover:rotate-0 duration-500">
                   <div className="w-full h-full border-2 border-black border-dashed rounded-xl flex items-center justify-center bg-white overflow-hidden relative">
                      <img src="/qrcode1.jpeg" alt="QR Code" className="w-full h-full object-contain" />
                      <div className="absolute inset-0 bg-black/70 text-white flex flex-col items-center justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Download size={22} />
                        <span className="text-[10px] font-semibold uppercase tracking-wider">Save contact</span>
                      </div>
                   </div>
                </div>
                <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 glass-chip text-ink text-[10px] px-3 py-1 rounded-full whitespace-nowrap">
                   Scan or tap for VCard
                </div>
             </button>

             <h1 className="text-5xl md:text-7xl font-bold text-ink mb-6 tracking-tight">
               Hello, I'm <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 to-amber-300">Bhavesh</span>
             </h1>
             <p className="text-muted text-lg md:text-xl max-w-2xl leading-relaxed">
               MSCS @ NYU Courant · Graduate Assistant @ NYUAD NY Office.<br/>
               I build <b className="text-ink">full-stack AI systems</b> — <b className="text-ink">Android</b>, <b className="text-ink">backend</b>, and <b className="text-ink">agentic ML</b> — for real impact across industries.
             </p>
             <p className="text-faint text-sm mt-6 max-w-md">
               Ask the AI below anything about my work — it speaks in my voice, grounded in my resume.
             </p>

             <div className="absolute bottom-40 animate-bounce text-faint">
                <ChevronDown size={24} />
             </div>
          </section>

          <div ref={refs.about}><AboutSection accents={accents} /></div>
          <div ref={refs.journey}><JourneySection accents={accents} /></div>
          <div ref={refs.experience}><ExperienceSection accents={accents} /></div>
          <div ref={refs.skills}><SkillsSection accents={accents} /></div>
          <div ref={refs.projects}><ProjectsSection accents={accents} /></div>
          <div ref={refs.leadership}><LeadershipSection accents={accents} /></div>
          <div ref={refs.publications}><PublicationSection accents={accents} /></div>
          <div ref={refs.contact}><ContactSection accents={accents} /></div>
        </div>

        {/* FLOATING CHAT — Dynamic Island */}
        <ChatIsland
          messages={messages}
          isTyping={isTyping}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          suggestionChips={suggestionChips}
          isScrolling={isScrolling}
          chatBottomRef={chatBottomRef}
        />

      </main>
    </div>
  );
};

/* ============================== SECTIONS ============================== */

const Section = ({ children, className = '', innerRef }) => (
  <section
    ref={innerRef}
    className={`min-h-screen h-auto w-full snap-start flex items-center justify-center p-6 py-20 md:py-24 relative ${className}`}
  >
    {children}
  </section>
);

const AboutSection = ({ accents }) => (
  <Section>
    <div className="max-w-5xl w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center relative z-10">
       <div className="space-y-6">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${accents.emerald}`}>
             <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
             Open to Summer 2026 Roles
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-ink leading-tight tracking-tight">
             Building AI for <br/>
             <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 to-amber-300">problems that matter.</span>
          </h2>
          <p className="text-muted text-base md:text-lg leading-relaxed">
            I'm a Master's student at <b className="text-ink">NYU Courant</b> (Class of '27) and a <b className="text-ink">Graduate Assistant at NYU Abu Dhabi's New York Office</b>.
            <br/><br/>
            My path went <b className="text-ink">Android → ML → domain-specific AI</b>. I've shipped production systems serving 2,600+ users at <b className="text-ink">TCS</b>, trained drug-target prediction models at <b className="text-ink">Synopsys</b>, and led Jetpack Compose workshops for 50+ developers as a Google <b className="text-ink">GDSC</b> lead.
            <br/><br/>
            I build for <b className="text-ink">impact</b>, not hype.
          </p>
       </div>

       <div className="flex justify-center perspective-1000">
          <div className="relative w-72 h-80 transition-transform duration-700 transform-style-3d group hover:rotate-y-180">
             <div className="absolute inset-0 backface-hidden rounded-3xl overflow-hidden border border-subtle/15 shadow-2xl bg-surface-2">
                 <div className="absolute inset-0 bg-gradient-to-tr from-accent/20 to-transparent"></div>
                 <img src="/panda.png" className="w-full h-full object-cover" alt="Avatar" />
             </div>
             <div className="absolute inset-0 backface-hidden rotate-y-180 rounded-3xl overflow-hidden border border-subtle/15 shadow-2xl bg-surface-2">
                 <img src="/profile_photo.png" className="w-full h-full object-cover" alt="Profile" />
             </div>
          </div>
       </div>
    </div>
  </Section>
);

const JourneySection = ({ accents }) => (
  <Section>
    <div className="max-w-4xl w-full">
      <SectionHeader icon={Map} color="text-amber-500">My Journey</SectionHeader>
      <div className="space-y-6">
        <JourneyStep
          era="Era I — Android"
          years="2020 – 2022"
          color={accents.emerald}
          text="Started building Android apps — Kotlin, Jetpack Compose, Material 3. Ran workshops for 50+ devs at Google GDSC. Shipped a campus map used by 200+ students. Hit a wall: I needed intelligent search."
        />
        <JourneyStep
          era="Era II — ML"
          years="2022 – 2024"
          color={accents.blue}
          text="Went deep on deep learning. Published a mammography transfer-learning paper at ICAIA '24 (Best Presenter). Interned at Synopsys building a drug-target binding model on 3.2M samples — 86% accuracy, 85% R&D acceleration."
        />
        <JourneyStep
          era="Era III — Domain-Specific AI"
          years="2024 – now"
          color={accents.rose}
          text="Production enterprise scale at TCS (Northern Trust): 2,600+ users, 26M+ weekly txns, 99.2% uptime, a GenAI QA pipeline that cut validation time 30%. Now at NYU Courant shipping agentic AI — VoiceGraph, GyBuddy, SignalFlow — where the agent isn't a gimmick, it's the product."
        />
      </div>
    </div>
  </Section>
);

const ExperienceSection = ({ accents }) => (
  <Section>
    <div className="max-w-4xl w-full">
       <SectionHeader icon={Briefcase} color="text-accent">Work Experience</SectionHeader>
       <div className="space-y-4">
          <WorkCard
             role="Graduate Assistant" company="NYU Abu Dhabi — New York Office" date="Feb 2026 – Present" color={accents.rose}
             desc="Academic Support Team at 19 Washington Square North. Supporting academic & cultural initiatives across NYU's global network; liaising with NYUAD leadership and visiting UAE government partners."
          />
          <WorkCard
             role="System Engineer" company="Tata Consultancy Services · Northern Trust" date="Jun 2024 – Jul 2025" color={accents.blue}
             desc="Built a Java/JDBC/PostgreSQL user-provisioning REST API serving 2,600+ users with 99.2% uptime across 26M+ weekly transactions for a US custodian bank. Shipped a LangChain + GPT-4 + Selenium data-validation pipeline that caught 15+ pre-prod issues and cut manual QA 30%."
          />
          <WorkCard
             role="Deep Learning Intern" company="Synopsys Inc. — Bioinformatics / Healthcare" date="Jul 2023 – Sep 2023" color={accents.purple}
             desc="PyTorch drug-target binding affinity model trained on 3.2M protein-drug samples → 86% accuracy, batched inference, and 85% R&D cycle acceleration (validation from weeks → 48 hours)."
          />
          <WorkCard
             role="Android Security Intern" company="eSec Forte Technologies" date="Jul 2022 – Sep 2022" color={accents.emerald}
             desc="Built a custom Android Auto security scanner in Kotlin (static + dynamic vulnerability analysis; 15+ data-exfil zero-days found). Shipped an enterprise Flutter POC to 100+ clients across 3 release cycles."
          />
          <WorkCard
             role="Android Lead · Compose Camp Facilitator" company="Google Developer Student Club — GGSIPU" date="Mar 2022 – Mar 2023" color={accents.amber}
             desc="Official Google Compose Camp facilitator. Taught Jetpack Compose to 50+ students across 12+ workshops; 25% retention improvement through weekly curriculum iteration. Campus map app adopted by 200+ students."
          />
       </div>
    </div>
  </Section>
);

const SkillsSection = ({ accents }) => (
  <Section>
    <div className="max-w-5xl w-full">
       <SectionHeader icon={Cpu} color="text-amber-500">Technical Arsenal</SectionHeader>
       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SkillCard title="Languages"       items={['Kotlin', 'Java', 'Python', 'TypeScript', 'JavaScript', 'SQL', 'Dart', 'C++']} color={accents.rose} />
          <SkillCard title="AI / ML"         items={['PyTorch', 'TensorFlow', 'LangChain', 'LangGraph', 'ChromaDB', 'Gemini Live API', 'OpenAI GPT-4', 'Hugging Face']} color={accents.blue} />
          <SkillCard title="Android & Mobile" items={['Android SDK', 'Jetpack Compose', 'Material 3', 'Coroutines', 'Retrofit', 'Room', 'React Native', 'Flutter']} color={accents.emerald} />
          <SkillCard title="Backend & APIs"  items={['FastAPI', 'Spring (basic)', 'REST', 'PostgreSQL', 'JDBC', 'WebSocket', 'Neo4j', 'OpenAPI']} color={accents.amber} />
          <SkillCard title="Frontend & Web"  items={['React', 'Next.js', 'Vite', 'Tailwind', 'Three.js', 'Expo']} color={accents.purple} />
          <SkillCard title="Infra & Tools"   items={['Docker', 'Git', 'GCP', 'Firebase', 'Cloud Run', 'Railway', 'Linux', 'Figma']} color={accents.rose} />
       </div>
    </div>
  </Section>
);

const ProjectsSection = ({ accents }) => (
  <Section>
     <div className="max-w-5xl w-full">
        <SectionHeader icon={Folder} color="text-blue-500">Featured Projects</SectionHeader>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
           <ProjectCard
             title="VoiceGraph"
             desc="Voice-first interactive knowledge graph. Gemini Live bidirectional audio over WebSocket drives a Three.js 3D graph (569 nodes, 882 edges). 3-phase extraction + agentic GraphRAG with 8 tools."
             tags={['Gemini Live', 'LangChain', 'Neo4j', 'Three.js', 'FastAPI']}
             color={accents.purple}
             link="https://voicegraph-802587268683.us-central1.run.app/"
           />
           <ProjectCard
             title="GyBuddy"
             desc="AI running buddy on React Native. Gemini 2.5 Flash Native Audio coaches you in real time via 16kHz↔24kHz PCM streaming. 10 agent tools, parametric routes, treadmill auto-detection."
             tags={['Gemini Live', 'React Native', 'Expo', 'Maps API', 'Firebase']}
             color={accents.amber}
             link="https://github.com/bhaveshgupta01"
           />
           <ProjectCard
             title="SignalFlow"
             desc="Autonomous AI crypto trading agent. 6 async triggers scan Polymarket + whale wallets + funding rates; Gemini evaluates with 85-tool Boba MCP. Overnight run: $100 → $105.56, 78% win rate."
             tags={['Agentic AI', 'Gemini Vertex', 'MCP', 'Hyperliquid', 'Streamlit']}
             color={accents.rose}
             link="https://github.com/bhaveshgupta01"
           />
           <ProjectCard
             title="Virtual War Room"
             desc="Top Technical Build at Google × Columbia (50+ teams). Multi-agent voice boardroom with 4 AI personas, raise-hand interjection, structured voting, real-time WebSocket audio."
             tags={['Gemini Live', 'Google ADK', 'FastAPI', 'React', 'WebSocket']}
             color={accents.blue}
             link="https://github.com/bhaveshgupta01"
           />
           <ProjectCard
             title="Pulse NYC — AI Marketing (1st Place)"
             desc="Event-triggered campaign deployment under 5 seconds. Real-time broadcast detection, targeted campaign generation, live prediction validation. 1st place winner."
             tags={['Next.js', 'TypeScript', 'AI', 'Real-time']}
             color={accents.emerald}
             link="https://github.com/bhaveshgupta01"
           />
           <ProjectCard
             title="Agentic RAG Engine"
             desc="Self-correcting RAG via LangGraph. Grades its own retrieval; falls back to web search if local docs are weak. Powers this portfolio's chat, re-grounded in my resume."
             tags={['LangGraph', 'ChromaDB', 'Gemini', 'Self-correction']}
             color={accents.purple}
             link="https://github.com/bhaveshgupta01"
           />
           <ProjectCard
             title="ICAIA '24 — Mammography Classification"
             desc="Bachelor's thesis. Deep transfer learning comparison for mammographic image classification. Published at ICAIA '24 with Best Presenter Award (DOI-indexed)."
             tags={['PyTorch', 'Transfer Learning', 'Healthcare AI']}
             color={accents.rose}
             link="https://doi.org/10.24874/PES.SI.25.03A.007"
           />
           <ProjectCard
             title="MoMA Quest"
             desc="PWA gamifying MoMA exploration. Gemini generates art-based quests from 200+ artworks; AI-generated runner personas, QR invites, weekly leaderboards, pack challenges."
             tags={['Next.js', 'Gemini', 'PWA', 'Firestore']}
             color={accents.amber}
             link="https://github.com/bhaveshgupta01"
           />
        </div>
     </div>
  </Section>
);

const LeadershipSection = ({ accents }) => (
  <Section>
     <div className="max-w-4xl w-full">
        <SectionHeader icon={Award} color="text-accent">Leadership & Recognition</SectionHeader>
        <div className="grid gap-4">
            <WorkCard role="Android Lead & Google Facilitator" company="Google Developer Student Club — GGSIPU" date="2022 – 2023" color={accents.emerald}
              desc="Taught 50+ students in Jetpack Compose. 12+ workshops, 25% retention improvement through weekly curriculum iteration."
            />
            <WorkCard role="Technical Festival Lead — Infoxpression" company="GGSIPU" date="2022 – 2024" color={accents.amber}
              desc="Led 30+ events across a 3-day technical festival."
            />
            <WorkCard role="Nominated Student Council Member" company="GGSIPU" date="2022 – 2024" color={accents.blue}
              desc="Founding committee member managing a 10,000+ alumni database."
            />
            <WorkCard role="Project Lead — 'Dor'" company="Enactus, GGSIPU" date="2022 – 2023" color={accents.rose}
              desc="Social entrepreneurship initiative."
            />
            <WorkCard role="Hackathon Wins" company="NYC" date="2025 – 2026" color={accents.purple}
              desc="1st Place @ Pulse NYC (AI Marketing) · Top Technical Build @ Google × Columbia (Virtual War Room)."
            />
        </div>
      </div>
  </Section>
);

const PublicationSection = ({ accents }) => (
  <Section>
     <div className="max-w-4xl w-full">
        <SectionHeader icon={BookOpen} color="text-violet-500">Publications</SectionHeader>
        <a
          href="https://doi.org/10.24874/PES.SI.25.03A.007"
          target="_blank"
          rel="noopener noreferrer"
          className="block p-6 bg-surface-2 rounded-2xl border border-subtle/15 hover:border-violet-500/30 transition-all group"
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-lg font-bold text-ink italic group-hover:text-violet-500 dark:group-hover:text-violet-300 transition-colors">"Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification"</p>
              <p className="text-muted mt-2">Proceedings on Engineering Sciences (PES) · ICAIA 2024 · <span className="text-violet-500 dark:text-violet-400">Best Presenter Award</span></p>
            </div>
            <ExternalLink size={18} className="text-faint group-hover:text-violet-500 shrink-0"/>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
              <span className={`text-xs px-2 py-1 rounded border ${accents.purple}`}>Transfer Learning</span>
              <span className={`text-xs px-2 py-1 rounded border ${accents.purple}`}>ResNet50</span>
              <span className={`text-xs px-2 py-1 rounded border ${accents.purple}`}>Medical Imaging</span>
              <span className={`text-xs px-2 py-1 rounded border ${accents.purple}`}>DOI-indexed</span>
          </div>
        </a>
     </div>
  </Section>
);

const ContactSection = () => (
  <Section className="mesh-bg">
     <div className="w-full max-w-3xl mx-auto space-y-10">
        <div className="text-center space-y-4">
          <h2 className="text-4xl md:text-6xl font-bold text-ink tracking-tight">Let's build something real.</h2>
          <p className="text-muted text-base md:text-xl max-w-2xl mx-auto">
              Open to Summer 2026 internships and collaborations on AI, mobile, and systems.
          </p>
        </div>

        {/* Business card — click anywhere or the button to download .vcf */}
        <motion.button
          type="button"
          onClick={downloadVCard}
          whileHover={{ y: -4 }}
          whileTap={{ scale: 0.99 }}
          transition={{ type: 'spring', stiffness: 240, damping: 24 }}
          className="group w-full glass-strong glass-edge rounded-3xl p-6 md:p-7 text-left relative overflow-hidden"
          aria-label="Download contact card"
        >
          {/* signature watermark — geometric logo peeking from the corner */}
          <LogoIcon
            aria-hidden
            className="absolute -right-6 -bottom-6 w-40 h-40 text-ink opacity-[0.10] dark:opacity-[0.18] pointer-events-none"
            strokeWidth={32}
          />
          <div className="flex items-start gap-5 relative z-10">
            {/* avatar */}
            <div className="w-16 h-16 md:w-20 md:h-20 rounded-2xl overflow-hidden flex-shrink-0 ring-1 ring-subtle/20 bg-surface-2">
              <img src="/profile_photo.png" alt="" className="w-full h-full object-cover" />
            </div>
            {/* identity */}
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <h3 className="text-xl md:text-2xl font-bold text-ink tracking-tight">Bhavesh Gupta</h3>
                <span className="text-[11px] uppercase tracking-wider font-bold text-accent">vCard</span>
              </div>
              <p className="text-muted text-[13px] md:text-sm mt-0.5">MSCS @ NYU Courant · Graduate Assistant @ NYUAD NY</p>
              <div className="text-faint text-[11px] md:text-xs mt-1">Building AI for problems that matter.</div>
            </div>
            {/* qr thumbnail — also hints at download */}
            <div className="hidden md:flex w-16 h-16 bg-white rounded-xl p-1 flex-shrink-0 ring-1 ring-subtle/20">
              <img src="/qrcode1.jpeg" alt="QR" className="w-full h-full object-contain" />
            </div>
          </div>

          {/* info grid */}
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 relative z-10">
            <ContactRow icon={Mail}  label="Email"    value="bg2896@nyu.edu" />
            <ContactRow icon={Phone} label="Phone"    value="+1 (201) 492-8876" />
            <ContactRow icon={MapPin} label="Location" value="Manhattan, NY" />
            <ContactRow icon={QrCode} label="Web"     value="libralpanda.vercel.app" />
          </div>

          {/* CTA */}
          <div className="mt-6 flex items-center justify-between gap-4 relative z-10">
            <div className="text-[11px] md:text-xs text-faint">
              Adds to your Contacts app on iOS, Android, macOS, Gmail & Outlook.
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-ink text-app text-xs font-bold group-hover:scale-105 transition-transform shrink-0">
              <Download size={14}/>
              Save Contact
            </div>
          </div>
        </motion.button>

        {/* secondary socials */}
        <div className="flex flex-wrap justify-center gap-3">
           <a href="mailto:bg2896@nyu.edu" className="glass-chip flex items-center gap-2 px-4 py-2.5 rounded-xl text-ink hover:text-accent transition-colors text-sm font-medium">
              <Mail size={14}/> Email
           </a>
           <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="glass-chip flex items-center gap-2 px-4 py-2.5 rounded-xl text-ink hover:text-accent transition-colors text-sm font-medium">
              <Linkedin size={14}/> LinkedIn
           </a>
           <a href="https://github.com/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="glass-chip flex items-center gap-2 px-4 py-2.5 rounded-xl text-ink hover:text-accent transition-colors text-sm font-medium">
              <Github size={14}/> GitHub
           </a>
        </div>
     </div>
  </Section>
);

const ContactRow = ({ icon: Icon, label, value }) => (
  <div className="flex items-center gap-3 p-3 rounded-xl bg-surface-2/60 border border-subtle/15">
    <div className="w-9 h-9 rounded-lg bg-accent/15 text-accent flex items-center justify-center shrink-0">
      <Icon size={14}/>
    </div>
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-faint font-bold">{label}</div>
      <div className="text-sm text-ink font-medium truncate">{value}</div>
    </div>
  </div>
);

/* ============================= HELPERS ============================= */

const SectionHeader = ({ icon: Icon, color, children }) => (
  <h3 className="text-3xl md:text-4xl font-bold text-ink mb-10 flex items-center gap-3 tracking-tight">
    <Icon className={color} size={28}/> {children}
  </h3>
);

const SidebarItem = ({ icon: Icon, label, onClick }) => (
  <button
    onClick={onClick}
    className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 mb-0.5 text-muted hover:bg-surface-2 hover:text-ink group"
  >
    <Icon size={16} className="group-hover:text-accent transition-colors"/>
    <span className="font-medium text-[13.5px]">{label}</span>
  </button>
);

const WorkCard = ({ role, company, date, desc, color }) => (
  <div className="bg-surface-2 p-5 md:p-6 rounded-2xl border border-subtle/15 hover:border-subtle/20 transition-all group hover:bg-surface-3">
     <div className="flex justify-between items-start mb-1.5 gap-3">
        <h4 className="text-[15px] md:text-lg font-bold text-ink group-hover:text-accent transition-colors leading-tight">{role}</h4>
        <span className={`text-[10px] md:text-xs px-2 py-1 rounded-md whitespace-nowrap border ${color}`}>{date}</span>
     </div>
     <p className="text-muted text-[13px] mb-2 font-medium">{company}</p>
     <p className="text-faint text-[13px] leading-relaxed">{desc}</p>
  </div>
);

const JourneyStep = ({ era, years, text, color }) => (
  <div className="bg-surface-2 p-5 md:p-6 rounded-2xl border border-subtle/15 hover:bg-surface-3 transition-colors">
    <div className="flex justify-between items-start mb-2 gap-3">
      <h4 className="text-lg font-bold text-ink tracking-tight">{era}</h4>
      <span className={`text-[10px] md:text-xs px-2 py-1 rounded-md whitespace-nowrap border ${color}`}>{years}</span>
    </div>
    <p className="text-muted text-[13.5px] leading-relaxed">{text}</p>
  </div>
);

const SkillCard = ({ title, items, color }) => (
  <div className="bg-surface-2 p-5 md:p-6 rounded-2xl border border-subtle/15 hover:bg-surface-3 transition-colors">
     <h4 className="text-base font-bold text-ink mb-3">{title}</h4>
     <div className="flex flex-wrap gap-1.5">
        {items.map(i => <span key={i} className={`text-xs px-2.5 py-1 rounded-full border ${color}`}>{i}</span>)}
     </div>
  </div>
);

const ProjectCard = ({ title, desc, tags, color, link }) => (
  <a
    href={link}
    target="_blank"
    rel="noopener noreferrer"
    className="block bg-surface-2 p-5 md:p-6 rounded-2xl border border-subtle/15 hover:-translate-y-1 transition-all cursor-pointer hover:shadow-2xl hover:shadow-accent/10 group relative overflow-hidden"
  >
     <div className="absolute top-5 right-5 opacity-0 group-hover:opacity-100 transition-opacity text-faint">
        <ExternalLink size={16} />
     </div>
     <div className={`absolute top-0 left-0 w-[2px] h-full ${color ? color.split(' ').find(c => c.startsWith('bg-')) || 'bg-blue-500' : 'bg-blue-500'}`}></div>
     <h4 className="text-base md:text-lg font-bold text-ink mb-1.5 group-hover:text-accent transition-colors pr-8 tracking-tight">
       {title}
     </h4>
     <p className="text-faint text-[13px] mb-3.5 leading-relaxed line-clamp-4">
       {desc}
     </p>
     <div className="flex flex-wrap gap-1.5">
        {tags.map(t => (
           <span key={t} className={`text-[10px] px-2 py-0.5 rounded border border-subtle/15 ${color ? color.split(' ')[0] : 'text-muted'}`}>
              {t}
           </span>
        ))}
     </div>
  </a>
);

export default Home;
