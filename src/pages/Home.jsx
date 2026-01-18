import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, Sparkles, User, Briefcase, Cpu, Folder, Palette, Mail, 
  Github, Linkedin, Globe, ChevronDown, Award, BookOpen, Terminal, 
  X, Zap, Layers, Database, Lock, Eye
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { ExternalLink } from 'lucide-react';

const Home = () => {
  const navigate = useNavigate();
  
  // --- STATE ---
  const [hasStarted, setHasStarted] = useState(false); 
  const [isTyping, setIsTyping] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  // --- REFS ---
  const heroRef = useRef(null);
  const aboutRef = useRef(null);
  const experienceRef = useRef(null);
  const skillsRef = useRef(null);
  const projectsRef = useRef(null);
  const leadershipRef = useRef(null);
  const contactRef = useRef(null);
  const chatBottomRef = useRef(null);

  // --- PASTEL PALETTE ---
  const colors = {
    rose: "text-rose-400 bg-rose-500/10 border-rose-500/20 hover:bg-rose-500/20",
    amber: "text-amber-300 bg-amber-400/10 border-amber-400/20 hover:bg-amber-400/20",
    emerald: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20 hover:bg-emerald-500/20",
    blue: "text-blue-400 bg-blue-500/10 border-blue-500/20 hover:bg-blue-500/20",
    purple: "text-violet-400 bg-violet-500/10 border-violet-500/20 hover:bg-violet-500/20",
  };

  const suggestionChips = [
    { label: 'About Me', cmd: 'about', icon: <User size={14}/> },
    { label: 'Experience', cmd: 'experience', icon: <Briefcase size={14}/> },
    { label: 'Projects', cmd: 'projects', icon: <Folder size={14}/> },
    { label: 'Tech Stack', cmd: 'skills', icon: <Cpu size={14}/> },
    { label: 'Design Assets', cmd: 'design', icon: <Palette size={14}/> },
    { label: 'Contact Info', cmd: 'contact', icon: <Mail size={14}/> },
    { label: 'Clear Chat', cmd: 'clear', icon: <X size={14}/> },
  ];

  // --- AUTO SCROLL CHAT ---
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // --- COMMAND PROCESSOR ---
  const handleSend = (text = input) => {
    if (!text.trim()) return;

    if (!hasStarted) setHasStarted(true);

    // CLEAR COMMAND LOGIC
    if (text.toLowerCase() === 'clear') {
        setMessages([]);
        setInput('');
        return;
    }
    
    // Add User Message
    const userMsg = { id: Date.now(), role: 'user', text: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    // AI Response Logic
    setTimeout(() => {
      const lower = text.toLowerCase();
      let response = { id: Date.now() + 1, role: 'bot', text: '', actions: [] };

      if (lower.includes('design')) {
        response.text = "Redirecting to Design Gallery...";
        navigate('/design');
      } else if (lower.includes('about') || lower.includes('who')) {
        response.text = "Here is my profile overview. MSCS @ NYU & Ex-TCS.";
        aboutRef.current?.scrollIntoView({ behavior: 'smooth' });
      } else if (lower.includes('exp') || lower.includes('work') || lower.includes('job')) {
        response.text = "Navigating to Work History (TCS, Synopsys, eSec Forte)...";
        experienceRef.current?.scrollIntoView({ behavior: 'smooth' });
      } else if (lower.includes('skill') || lower.includes('stack')) {
        response.text = "Analyzing Technical Stack (Python, Java, AWS, Docker)...";
        skillsRef.current?.scrollIntoView({ behavior: 'smooth' });
      } else if (lower.includes('project')) {
        response.text = "Opening Project Archives (RAG Engine, Split-DNN)...";
        projectsRef.current?.scrollIntoView({ behavior: 'smooth' });
      } else if (lower.includes('contact') || lower.includes('email')) {
        response.text = "Scrolling to Contact Info...";
        contactRef.current?.scrollIntoView({ behavior: 'smooth' });
      } else if (lower.includes('gogo')) {
        response.text = "I do too <3";
      }else {
        response.text = "I can take you anywhere. Where would you like to go?";
      }

      setIsTyping(false);
      setMessages(prev => [...prev, response]);
    }, 800);
  };

  return (
    <div className="flex h-screen w-full bg-[#0a0a0a] text-slate-200 font-sans overflow-hidden relative selection:bg-rose-500/30">
      
      {/* --- SIDEBAR (Hidden on Mobile) --- */}
      <aside className="w-64 flex-shrink-0 bg-[#121212] border-r border-white/5 flex flex-col z-20 h-full hidden md:flex">
        <div className="p-6 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-400 to-amber-300 flex items-center justify-center shadow-lg shadow-rose-500/20 text-black font-bold">B</div>
          <span className="font-bold text-lg tracking-wide text-white">
            Portfolio<span className="text-rose-400">.ai</span>
          </span>
        </div>
        <nav className="flex-1 px-4 space-y-2 mt-4 overflow-y-auto">
          <SidebarItem icon={<User size={18}/>} label="About Me" onClick={() => aboutRef.current?.scrollIntoView({behavior:'smooth'})} />
          <SidebarItem icon={<Briefcase size={18}/>} label="Experience" onClick={() => experienceRef.current?.scrollIntoView({behavior:'smooth'})} />
          <SidebarItem icon={<Cpu size={18}/>} label="Tech Stack" onClick={() => skillsRef.current?.scrollIntoView({behavior:'smooth'})} />
          <SidebarItem icon={<Folder size={18}/>} label="Projects" onClick={() => projectsRef.current?.scrollIntoView({behavior:'smooth'})} />
          <SidebarItem icon={<Award size={18}/>} label="Leadership" onClick={() => leadershipRef.current?.scrollIntoView({behavior:'smooth'})} />
          <SidebarItem icon={<Palette size={18}/>} label="Designs" onClick={() => navigate('/design')} />
          <SidebarItem icon={<Mail size={18}/>} label="Contact" onClick={() => contactRef.current?.scrollIntoView({behavior:'smooth'})} />
        </nav>
      </aside>

      {/* --- MAIN SCROLLABLE CONTENT --- */}
      <main className="flex-1 relative h-full w-full">
        <div className="absolute inset-0 overflow-y-auto scroll-smooth snap-y snap-mandatory pb-48 z-0" id="main-scroll">
            
            {/* HERO SECTION */}
            <section ref={heroRef} className="h-screen w-full snap-start flex flex-col items-center justify-center p-6 text-center relative bg-gradient-to-b from-[#0a0a0a] to-[#111]">
                 <div className="mb-8 relative group cursor-pointer" onClick={() => contactRef.current?.scrollIntoView({behavior:'smooth'})}>
                    <div className="absolute inset-0 bg-rose-500 blur-[60px] opacity-20 rounded-full group-hover:opacity-40 transition-opacity"></div>
                    <div className="w-40 h-40 bg-white p-2 rounded-2xl shadow-2xl rotate-3 transition-transform group-hover:rotate-0 duration-500">
                       <div className="w-full h-full border-2 border-black border-dashed rounded-xl flex items-center justify-center bg-white overflow-hidden">
                          {/* ⚠️ REPLACE THIS WITH YOUR QR CODE IMAGE */}
                          <img src="/qrcode1.jpeg" alt="QR Code" className="w-full h-full object-contain" />
                       </div>
                    </div>
                    <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 bg-[#1a1a1a] text-white text-[10px] px-3 py-1 rounded-full border border-white/10 whitespace-nowrap shadow-xl">
                       Scan for VCard
                    </div>
                 </div>
                 
                 <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
                   Hello, I'm <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 to-amber-300">Bhavesh</span>
                 </h1>
                 <p className="text-slate-400 text-lg md:text-xl max-w-2xl leading-relaxed">
                   MSCS @ NYU Courant. Specializing in <b>Distributed Systems</b>, <b>HPC</b>, and <b>Agentic AI</b>.<br/>
                   Explore my work below or ask the AI assistant.
                 </p>
                 
                 <div className="absolute bottom-40 animate-bounce text-slate-500">
                    <ChevronDown size={24} />
                 </div>
            </section>

            {/* ABOUT SECTION (With 3D Flip) */}
            <div ref={aboutRef}><AboutSection colors={colors} /></div>

            {/* EXPERIENCE SECTION */}
            <div ref={experienceRef}><ExperienceSection colors={colors} /></div>

            {/* SKILLS SECTION */}
            <div ref={skillsRef}><SkillsSection colors={colors} /></div>

            {/* PROJECTS SECTION */}
            <div ref={projectsRef}><ProjectsSection colors={colors} /></div>

             {/* LEADERSHIP SECTION */}
             <div ref={leadershipRef}><LeadershipSection colors={colors} /></div>

             {/* PUBLICATION SECTION */}
            <div ref={experienceRef}><PublicationSection colors={colors} /></div>

            {/* CONTACT SECTION */}
            <div ref={contactRef}><ContactSection colors={colors} /></div>
        </div>

        {/* --- FLOATING CHAT INTERFACE --- */}
        <div className="absolute bottom-0 left-0 right-0 z-50 pointer-events-none p-4 md:p-8 flex flex-col items-center justify-end h-screen">
            
            {/* Messages Container */}
            <div className={`w-full max-w-3xl transition-all duration-500 ease-out mb-2 ${
                hasStarted ? 'opacity-100 translate-y-0 max-h-[60vh]' : 'opacity-0 translate-y-10 h-0'
            } overflow-y-auto scrollbar-hide pointer-events-auto flex flex-col justify-end mask-image-linear-gradient`}>
               
               <div className="space-y-3 pb-2">
                   {messages.map((msg) => (
                      <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
                          <div className={`px-5 py-3 rounded-2xl max-w-[85%] backdrop-blur-md shadow-lg border ${
                              msg.role === 'user' 
                              ? 'bg-[#1a1a1a] text-white border-white/10' 
                              : 'bg-[#1a1a1a] text-slate-200 border-rose-500/20'
                          }`}>
                              <p className="text-sm md:text-base leading-relaxed">{msg.text}</p>
                          </div>
                      </div>
                   ))}
                   {isTyping && (
                       <div className="flex justify-start animate-fade-in">
                           <div className="bg-[#1a1a1a] px-4 py-3 rounded-2xl border border-white/10 flex gap-1.5 shadow-lg">
                               <div className="w-1.5 h-1.5 bg-rose-400 rounded-full animate-bounce"></div>
                               <div className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                               <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                           </div>
                       </div>
                   )}
                   <div ref={chatBottomRef} />
               </div>
            </div>

            {/* Persistent Suggestions Carousel */}
            <div className="w-full max-w-3xl pointer-events-auto mb-3 overflow-x-auto scrollbar-hide">
                <div className="flex gap-2 whitespace-nowrap px-1">
                    {suggestionChips.map((chip, idx) => (
                        <button
                            key={idx}
                            onClick={() => handleSend(chip.cmd)}
                            className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#1E1E1E]/90 hover:bg-[#2a2a2a] border border-white/10 text-xs font-medium text-slate-300 transition-all hover:scale-105 hover:border-rose-500/30 shadow-lg backdrop-blur-sm"
                        >
                            {chip.icon}
                            {chip.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Input Bar */}
            <div className="w-full max-w-3xl pointer-events-auto">
                <div className="bg-[#1a1a1a]/80 backdrop-blur-xl border border-white/10 p-2 rounded-2xl shadow-2xl flex items-center gap-2 ring-1 ring-white/5 focus-within:ring-rose-500/50 transition-all">
                    <div className="p-3 bg-white/5 rounded-xl text-rose-400">
                        <Sparkles size={20} />
                    </div>
                    <input 
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask about projects, skills, or experience..."
                        className="flex-1 bg-transparent border-none outline-none text-white placeholder-slate-500 h-10 px-2 font-medium"
                    />
                    <button 
                        onClick={() => handleSend()}
                        className="p-3 bg-white text-black rounded-xl hover:bg-rose-50 transition-colors shadow-lg shadow-rose-500/20"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </div>
        </div>

      </main>
    </div>
  );
};

/* --- SECTIONS WITH REAL DATA --- */

const AboutSection = ({ colors }) => (
  <section className="h-screen w-full snap-start flex items-center justify-center p-6 relative bg-[#0a0a0a] overflow-hidden">
    <div className="max-w-5xl w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center relative z-10">
       <div className="space-y-6">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${colors.emerald}`}>
             <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
             Open to Opportunities
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-white leading-tight">
             Forging Logic into <br/>
             <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 to-amber-300">Digital Reality.</span>
          </h2>
          <p className="text-slate-400 text-lg leading-relaxed">
             I am a Master's student at <b>NYU Courant</b> (Class of '27) with a proven track record at <b>TCS</b> and <b>Synopsys</b>.
             <br/><br/>
             My expertise lies in building privacy-preserving AI frameworks, high-frequency trading engines, and scalable distributed systems.
          </p>
       </div>
       
       {/* 3D PHOTO TWIST */}
       <div className="flex justify-center perspective-1000">
          <div className="relative w-72 h-80 transition-transform duration-700 transform-style-3d group hover:rotate-y-180">
             {/* Front Side */}
             <div className="absolute inset-0 backface-hidden rounded-3xl overflow-hidden border border-white/10 shadow-2xl bg-[#1E1E1E]">
                 <div className="absolute inset-0 bg-gradient-to-tr from-rose-500/20 to-transparent"></div>
                 <img src="/panda.png" className="w-full h-full object-cover" alt="Avatar" />
             </div>
             {/* Back Side */}
             <div className="absolute inset-0 backface-hidden rotate-y-180 rounded-3xl overflow-hidden border border-white/10 shadow-2xl bg-[#1E1E1E]">
                 <img src="/profile_photo.png" className="w-full h-full object-cover" alt="Profile" />
             </div>
          </div>
       </div>
    </div>
  </section>
);

const ExperienceSection = ({ colors }) => (
  <section className="h-screen w-full snap-start flex items-center justify-center p-6 bg-[#0c0c0c]">
    <div className="max-w-4xl w-full">
       <h3 className="text-3xl font-bold text-white mb-10 flex items-center gap-3">
         <Briefcase className="text-rose-400"/> Work Experience
       </h3>
       <div className="space-y-6">
          <WorkCard 
             role="System Engineer" company="Tata Consultancy Services" date="June 2024 - July 2025" color={colors.blue}
             desc="Architected a Java/SQL financial trading system for Northern Trust handling 26M+ weekly transactions. Developed GenAI Python scripts for data integrity validation." 
          />
          <WorkCard 
             role="ML/AI Research Intern" company="Synopsys Inc." date="July 2023 - Sept 2023" color={colors.purple}
             desc="Engineered a Deep Learning model with 98% accuracy for drug-target binding prediction. Reduced R&D timelines by 90% via SQL/Python optimization." 
          />
          <WorkCard 
             role="Android Dev Intern" company="eSec Forte" date="May 2022 - July 2022" color={colors.emerald}
             desc="Developed 'BunKey', a secure E2E encrypted Android chat app using Kotlin and custom dual-key ciphers. Conducted penetration testing." 
          />
       </div>
    </div>
  </section>
);

const SkillsSection = ({ colors }) => (
  // FIXED: Added 'min-h-screen h-auto py-20' for mobile responsiveness
  <section className="min-h-screen h-auto w-full snap-start flex items-center justify-center p-6 py-20 bg-[#0a0a0a] relative z-10">
    <div className="max-w-5xl w-full">
       <h3 className="text-3xl font-bold text-white mb-10 flex items-center gap-3">
         <Cpu className="text-amber-400"/> Technical Arsenal
       </h3>
       
       <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Languages */}
          <SkillCard 
            title="Languages" 
            items={['Python', 'Java', 'C++', 'SQL', 'Kotlin', 'JavaScript', 'HTML/CSS', 'MATLAB']} 
            color={colors.rose} 
          />
          
          {/* Machine Learning & Data */}
          <SkillCard 
            title="ML & Data Science" 
            items={['PyTorch', 'TensorFlow', 'Pandas', 'NumPy', 'SciPy', 'Hugging Face', 'Scikit-learn', 'LangGraph']} 
            color={colors.blue} 
          />
          
          {/* Infrastructure & Cloud */}
          <SkillCard 
            title="Infrastructure" 
            items={['AWS (EC2/S3)', 'Docker', 'Kubernetes', 'Git', 'Linux', 'REST APIs', 'CI/CD']} 
            color={colors.amber} 
          />
          
          {/* Web & Tools */}
          <SkillCard 
            title="Web & Tools" 
            items={['React', 'Spring Boot', 'FastAPI', 'Figma', 'Canva', 'Asana', 'Jira']} 
            color={colors.emerald} 
          />
       </div>
    </div>
  </section>
);

const ProjectsSection = ({ colors }) => (
  // FIXED: 'min-h-screen h-auto py-20' prevents overflow on mobile
  <section className="min-h-screen h-auto w-full snap-start flex items-center justify-center p-6 py-20 bg-[#0c0c0c] relative z-10">
     <div className="max-w-5xl w-full">
        <h3 className="text-3xl font-bold text-white mb-10 flex items-center gap-3">
          <Folder className="text-blue-400"/> Featured Projects
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
           {/* 1. RAG Engine */}
           <ProjectCard 
             title="Vision-Agentic RAG Engine" 
             desc="Multi-modal system using LLaVA + Mistral. Implements self-correction loops and OCR for document understanding." 
             tags={['LangGraph', 'ChromaDB', 'LLMs']} 
             color={colors.purple}
             link="https://github.com/bhaveshgupta01" // <--- Add URL here
           />

           {/* 2. Split-DNN */}
           <ProjectCard 
             title="Split-DNN (Privacy AI)" 
             desc="Quantized Deep Neural Network pipeline on edge devices. Achieved 3.7x compression and +10dB privacy defense." 
             tags={['PyTorch', 'Edge AI', 'Python']} 
             color={colors.amber}
             link="https://github.com/bhaveshgupta01" // <--- Add URL here
           />

           {/* 3. Trading Engine (No link provided example) */}
           <ProjectCard 
             title="High-Freq Trading Engine" 
             desc="Java-based matching engine for Northern Trust client. Optimized for low-latency processing of 26M+ txns." 
             tags={['Java', 'SQL', 'FinTech']} 
             color={colors.blue}
             link="#" // <--- Leave # or remove prop if private
           />

           {/* 4. Portfolio */}
           <ProjectCard 
             title="Dockerized Portfolio" 
             desc="Containerized React/Python terminal app with live system metrics via Flask/FastAPI." 
             tags={['Docker', 'React', 'AWS']} 
             color={colors.rose}
             link="https://github.com/bhaveshgupta01" // <--- Add URL here
           />
        </div>
     </div>
  </section>
);

const LeadershipSection = ({ colors }) => (
  <section className="h-screen w-full snap-start flex items-center justify-center p-6 bg-[#0a0a0a]">
     <div className="max-w-4xl w-full space-y-12">
        <div>
            <h3 className="text-3xl font-bold text-white mb-8 flex items-center gap-3">
                <Award className="text-rose-400"/> Leadership
            </h3>
            <div className="grid gap-4">
                <WorkCard role="Placement Coordinator" company="GGSIPU" date="2022-24" color={colors.blue} desc="Managed recruitment for 600+ students. Secured 120+ offers from Airtel, Jio, Samsung." />
                <WorkCard role="Android Lead" company="Google Developer Student Club" date="2022-24" color={colors.emerald} desc="Taught 50+ students in Android Dev. Created reusable educational materials." />
                <WorkCard role="President" company="University Science Club" date="2022-24" color={colors.amber} desc="Organized InfoXpression '23 TechFest for 10,000+ attendees." />
            </div>
        </div>
      </div>
  </section>
);

const PublicationSection = ({ colors }) => (
  <section className="h-screen w-full snap-start flex items-center justify-center p-6 bg-[#0a0a0a]">
     <div className="max-w-4xl w-full space-y-12">
        <div>
            <h3 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <BookOpen className="text-purple-400"/> Publications
            </h3>
            <div className="p-6 bg-[#1a1a1a] rounded-xl border border-white/5">
                <p className="text-lg font-bold text-white italic">"Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification"</p>
                <p className="text-slate-400 mt-2">Proceedings on Engineering Sciences (PES), Conference: ICAIA 2024</p>
                <div className="mt-4 flex gap-2">
                    <span className={`text-xs px-2 py-1 rounded border ${colors.purple}`}>ResNet50</span>
                    <span className={`text-xs px-2 py-1 rounded border ${colors.purple}`}>Medical AI</span>
                </div>
            </div>
        </div>
     </div>
  </section>
);

const ContactSection = ({ colors }) => (
  <section className="h-screen w-full snap-start flex items-center justify-center p-6 bg-[#0c0c0c]">
     <div className="text-center space-y-10">
        <h2 className="text-6xl font-bold text-white">Let's Build the Future.</h2>
        <p className="text-slate-400 text-xl max-w-xl mx-auto">
            Open to Summer 2026 Internships & Collaborations.
        </p>
        <div className="flex flex-wrap justify-center gap-6">
           <a href="mailto:bg2896@nyu.edu" className="flex items-center gap-2 px-8 py-4 bg-[#1E1E1E] rounded-2xl text-white hover:bg-rose-600 transition-all border border-white/10 hover:scale-105">
              <Mail/> Email Me
           </a>
           <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" className="flex items-center gap-2 px-8 py-4 bg-[#1E1E1E] rounded-2xl text-white hover:bg-[#0077b5] transition-all border border-white/10 hover:scale-105">
              <Linkedin/> LinkedIn
           </a>
           <a href="https://github.com/bhaveshgupta01" target="_blank" className="flex items-center gap-2 px-8 py-4 bg-[#1E1E1E] rounded-2xl text-white hover:bg-slate-700 transition-all border border-white/10 hover:scale-105">
              <Github/> GitHub
           </a>
        </div>
     </div>
  </section>
);

/* --- HELPER COMPONENTS --- */
const SidebarItem = ({ icon, label, onClick }) => (
  <button onClick={onClick} className="w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 mb-1 text-slate-500 hover:bg-white/5 hover:text-slate-200">
    {React.cloneElement(icon, { size: 18 })}
    <span className="font-medium text-sm">{label}</span>
  </button>
);

const WorkCard = ({ role, company, date, desc, color }) => (
  <div className="bg-[#1a1a1a] p-6 rounded-2xl border border-white/5 hover:border-white/10 transition-all group hover:bg-[#202020]">
     <div className="flex justify-between items-start mb-2">
        <h4 className="text-lg md:text-xl font-bold text-white group-hover:text-rose-400 transition-colors">{role}</h4>
        <span className={`text-[10px] md:text-xs px-2 py-1 rounded-md whitespace-nowrap ${color}`}>{date}</span>
     </div>
     <p className="text-slate-400 text-sm mb-2 font-medium">{company}</p>
     <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
  </div>
);

const SkillCard = ({ title, items, color }) => (
  <div className="bg-[#1a1a1a] p-6 rounded-2xl border border-white/5 hover:bg-[#202020] transition-colors">
     <h4 className="text-lg font-bold text-white mb-4">{title}</h4>
     <div className="flex flex-wrap gap-2">
        {items.map(i => <span key={i} className={`text-xs px-3 py-1.5 rounded-full ${color}`}>{i}</span>)}
     </div>
  </div>
);

const ProjectCard = ({ title, desc, tags, color, link }) => (
  <a 
    href={link}
    target="_blank"
    rel="noopener noreferrer"
    className="block bg-[#1a1a1a] p-6 rounded-2xl border border-white/5 hover:-translate-y-1 transition-all cursor-pointer hover:shadow-2xl hover:shadow-purple-500/10 group relative overflow-hidden"
  >
     {/* Hover Link Icon */}
     <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400">
        <ExternalLink size={18} />
     </div>

     {/* Decorative Line */}
     <div className={`absolute top-0 left-0 w-1 h-full ${color ? color.replace('text-', 'bg-').split(' ')[1] : 'bg-blue-500'}`}></div>

     <h4 className="text-lg font-bold text-white mb-2 group-hover:text-purple-400 transition-colors pr-8">
       {title}
     </h4>
     <p className="text-slate-500 text-sm mb-4 leading-relaxed line-clamp-3">
       {desc}
     </p>
     <div className="flex flex-wrap gap-2">
        {tags.map(t => (
           <span key={t} className={`text-[10px] px-2 py-1 rounded border border-white/10 ${color ? color.split(' ')[0] : 'text-slate-400'}`}>
              {t}
           </span>
        ))}
     </div>
  </a>
);

export default Home;