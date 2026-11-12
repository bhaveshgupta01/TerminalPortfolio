import React from 'react';
import {
  User,
  Briefcase,
  Cpu,
  Folder,
  Palette,
  Mail,
  Github,
  Linkedin,
  Map,
  Award,
  BookOpen,
  ExternalLink,
  Download,
  Phone,
  MapPin,
  QrCode,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import LogoIcon from '../LogoIcon';
import { downloadVCard } from '../../lib/vcard';
import { designs } from '../../data/designs';

const accents = {
  rose: 'text-rose-500 bg-rose-500/10 border-rose-500/20',
  amber: 'text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20',
  emerald: 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  blue: 'text-blue-600 dark:text-blue-400 bg-blue-500/10 border-blue-500/20',
  purple: 'text-violet-600 dark:text-violet-400 bg-violet-500/10 border-violet-500/20',
};

const featuredDesigns = designs.slice(0, 4);
const designCounts = Object.entries(
  designs.reduce((acc, design) => {
    acc[design.category] = (acc[design.category] || 0) + 1;
    return acc;
  }, {}),
).map(([category, count]) => ({ category, count }));

const SectionHeading = ({ icon: Icon, color, title, kicker, children }) => (
  <div className="space-y-3">
    <div className="flex flex-wrap items-center gap-3">
      <div className={`flex h-11 w-11 items-center justify-center rounded-2xl border ${color}`}>
        <Icon size={18} />
      </div>
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-faint">
          {kicker}
        </div>
        <h3 className="text-2xl font-bold tracking-tight text-ink">{title}</h3>
      </div>
    </div>
    {children && <p className="max-w-3xl text-sm leading-relaxed text-muted">{children}</p>}
  </div>
);

const StatTile = ({ label, value, sub }) => (
  <div className="rounded-2xl border border-subtle/15 bg-surface-2/70 p-4">
    <div className="text-lg font-bold text-ink">{value}</div>
    <div className="mt-1 text-xs font-medium uppercase tracking-[0.22em] text-faint">{label}</div>
    {sub && <div className="mt-2 text-xs leading-relaxed text-muted">{sub}</div>}
  </div>
);

const WorkCard = ({ role, company, date, desc, color }) => (
  <div className="rounded-2xl border border-subtle/15 bg-surface-2/65 p-5">
    <div className="flex items-start justify-between gap-3">
      <div>
        <h4 className="text-base font-bold tracking-tight text-ink">{role}</h4>
        <p className="mt-1 text-sm font-medium text-muted">{company}</p>
      </div>
      <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${color}`}>
        {date}
      </span>
    </div>
    <p className="mt-4 text-[13px] leading-relaxed text-faint">{desc}</p>
  </div>
);

const JourneyStep = ({ era, years, text, color }) => (
  <div className="rounded-2xl border border-subtle/15 bg-surface-2/65 p-5">
    <div className="flex items-start justify-between gap-3">
      <div className="text-lg font-bold tracking-tight text-ink">{era}</div>
      <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${color}`}>
        {years}
      </span>
    </div>
    <p className="mt-3 text-[13px] leading-relaxed text-muted">{text}</p>
  </div>
);

const SkillCard = ({ title, items, color }) => (
  <div className="rounded-2xl border border-subtle/15 bg-surface-2/65 p-5">
    <h4 className="text-base font-bold tracking-tight text-ink">{title}</h4>
    <div className="mt-4 flex flex-wrap gap-2">
      {items.map((item) => (
        <span key={item} className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${color}`}>
          {item}
        </span>
      ))}
    </div>
  </div>
);

const ProjectCard = ({ title, desc, tags, color, link }) => (
  <a
    href={link}
    target="_blank"
    rel="noopener noreferrer"
    className="group block rounded-2xl border border-subtle/15 bg-surface-2/65 p-5 transition-colors hover:border-accent/25"
  >
    <div className="flex items-start justify-between gap-4">
      <h4 className="text-base font-bold tracking-tight text-ink group-hover:text-accent">{title}</h4>
      <ExternalLink size={16} className="shrink-0 text-faint transition-colors group-hover:text-accent" />
    </div>
    <p className="mt-3 text-[13px] leading-relaxed text-muted">{desc}</p>
    <div className="mt-4 flex flex-wrap gap-2">
      {tags.map((tag) => (
        <span key={tag} className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${color}`}>
          {tag}
        </span>
      ))}
    </div>
  </a>
);

const ContactRow = ({ icon: Icon, label, value }) => (
  <div className="flex items-center gap-3 rounded-2xl border border-subtle/15 bg-surface-2/60 p-3.5">
    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/12 text-accent">
      <Icon size={15} />
    </div>
    <div className="min-w-0">
      <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-faint">{label}</div>
      <div className="truncate text-sm font-medium text-ink">{value}</div>
    </div>
  </div>
);

const AboutContent = () => (
  <div className="space-y-7">
    <SectionHeading icon={User} color={accents.rose} title="About Me" kicker="Identity">
      The wallet version keeps the same portfolio story, but compresses it into an interface where the AI can surface the right card instead of asking you to scroll through a landing page.
    </SectionHeading>

    <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
      <div className="space-y-5 rounded-[28px] border border-subtle/15 bg-surface-2/60 p-6">
        <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${accents.emerald}`}>
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          Open to Summer 2026 roles
        </div>

        <div>
          <h4 className="text-3xl font-bold tracking-tight text-ink">
            Building AI for
            {' '}
            <span className="bg-gradient-to-r from-rose-400 to-amber-300 bg-clip-text text-transparent">
              problems that matter
            </span>
          </h4>
          <p className="mt-4 text-sm leading-relaxed text-muted">
            I&apos;m a Master&apos;s student at NYU Courant and a Graduate Assistant at NYU Abu Dhabi&apos;s New York Office.
            My path went Android to ML to domain-specific AI, and the common thread is shipping work that is useful in the real world.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-muted">
            I&apos;ve shipped enterprise systems for Northern Trust at TCS, built healthcare AI at Synopsys, taught Android at Google Developer Student Club, and now build agentic experiences where the interface is part of the product.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <StatTile label="Current Base" value="NYC" sub="MSCS at NYU Courant + NYUAD NY Office" />
          <StatTile label="Core Stack" value="AI + Mobile" sub="React, FastAPI, LangGraph, Gemini, Android" />
          <StatTile label="Operating Mode" value="Impact" sub="Production systems, not portfolio theater" />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
        <div className="relative overflow-hidden rounded-[28px] border border-subtle/15 bg-surface-2/70 p-5">
          <LogoIcon
            aria-hidden
            className="absolute -right-5 -top-5 h-28 w-28 text-ink opacity-10"
            strokeWidth={30}
          />
          <div className="relative z-10 flex items-start gap-4">
            <div className="h-20 w-20 overflow-hidden rounded-2xl bg-surface ring-1 ring-subtle/15">
              <img src="/profile_photo.png" alt="Bhavesh Gupta" className="h-full w-full object-cover" />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.22em] text-faint">Current Focus</div>
              <div className="mt-2 text-lg font-bold tracking-tight text-ink">Agentic product systems</div>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                Voice interfaces, tool-using agents, structured UI responses, and interfaces that react to what the model is doing.
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-[28px] border border-subtle/15 bg-surface-2/70 p-5">
          <div className="text-xs font-semibold uppercase tracking-[0.22em] text-faint">What matters to me</div>
          <div className="mt-4 flex flex-wrap gap-2">
            {['Product sense', 'Systems thinking', 'Applied AI', 'Mobile UX', 'Real-world metrics'].map((item) => (
              <span key={item} className="rounded-full border border-subtle/15 px-3 py-1.5 text-xs font-medium text-ink">
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
);

const JourneyContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Map} color={accents.amber} title="My Journey" kicker="Trajectory">
      The wallet stack works well here because the story is sequential. Each era built the constraints for the next one.
    </SectionHeading>

    <div className="space-y-4">
      <JourneyStep
        era="Era I — Android"
        years="2020 – 2022"
        color={accents.emerald}
        text="Started building Android apps with Kotlin, Jetpack Compose, and Material 3. Ran workshops for 50+ developers and shipped a campus map app used by 200+ students."
      />
      <JourneyStep
        era="Era II — ML"
        years="2022 – 2024"
        color={accents.blue}
        text="Moved deeper into machine learning and healthcare AI. Published mammography transfer-learning research and built a drug-target binding model at Synopsys on 3.2M protein-drug samples."
      />
      <JourneyStep
        era="Era III — Domain-Specific AI"
        years="2024 – now"
        color={accents.rose}
        text="At TCS I shipped enterprise systems at real scale. Now at NYU I’m building agentic software where model orchestration, live tools, and UI state all have to feel coherent together."
      />
    </div>
  </div>
);

const ExperienceContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Briefcase} color={accents.blue} title="Experience" kicker="Work">
      Enterprise backend delivery, research-heavy AI work, and developer education all sit in the same timeline here.
    </SectionHeading>

    <div className="space-y-4">
      <WorkCard
        role="Graduate Assistant"
        company="NYU Abu Dhabi — New York Office"
        date="2026 – Present"
        color={accents.rose}
        desc="Supporting academic and cultural initiatives across NYU’s global network, while coordinating with leadership and visiting government partners."
      />
      <WorkCard
        role="System Engineer"
        company="Tata Consultancy Services · Northern Trust"
        date="2024 – 2025"
        color={accents.blue}
        desc="Built a Java, JDBC, and PostgreSQL user-provisioning REST API serving 2,600+ users across 26M+ weekly transactions with 99.2% uptime, and shipped a GenAI data-validation pipeline that cut QA effort 30%."
      />
      <WorkCard
        role="Deep Learning Intern"
        company="Synopsys Inc. — Bioinformatics / Healthcare"
        date="2023"
        color={accents.purple}
        desc="Trained a PyTorch drug-target binding model over 3.2M protein-drug samples, reaching 86% accuracy and accelerating research validation from weeks to 48 hours."
      />
      <WorkCard
        role="Android Security Intern"
        company="eSec Forte Technologies"
        date="2022"
        color={accents.emerald}
        desc="Built a custom Android Auto security scanner, uncovered 15+ zero-day style vulnerabilities, and shipped a Flutter enterprise proof of concept across multiple release cycles."
      />
    </div>
  </div>
);

const SkillsContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Cpu} color={accents.amber} title="Technical Arsenal" kicker="Stack">
      The skills card stays broad because the backend can already narrow it via `scroll_to`, and later via streamed UI commands.
    </SectionHeading>

    <div className="grid gap-4 lg:grid-cols-2">
      <SkillCard title="Languages" items={['Kotlin', 'Java', 'Python', 'TypeScript', 'JavaScript', 'SQL', 'Dart', 'C++']} color={accents.rose} />
      <SkillCard title="AI / ML" items={['PyTorch', 'TensorFlow', 'LangChain', 'LangGraph', 'ChromaDB', 'Gemini Live API', 'OpenAI GPT-4', 'Hugging Face']} color={accents.blue} />
      <SkillCard title="Android & Mobile" items={['Android SDK', 'Jetpack Compose', 'Material 3', 'Coroutines', 'Retrofit', 'Room', 'React Native', 'Flutter']} color={accents.emerald} />
      <SkillCard title="Backend & APIs" items={['FastAPI', 'Spring (basic)', 'REST', 'PostgreSQL', 'JDBC', 'WebSocket', 'Neo4j', 'OpenAPI']} color={accents.amber} />
      <SkillCard title="Frontend & Web" items={['React', 'Next.js', 'Vite', 'Tailwind', 'Three.js', 'Expo']} color={accents.purple} />
      <SkillCard title="Infra & Tools" items={['Docker', 'Git', 'GCP', 'Firebase', 'Cloud Run', 'Railway', 'Linux', 'Figma']} color={accents.rose} />
    </div>
  </div>
);

const ProjectsContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Folder} color={accents.blue} title="Featured Projects" kicker="Builds">
      This card is likely to be the first one the AI focuses once streaming commands are added, so the previews and links are already structured for that.
    </SectionHeading>

    <div className="grid gap-4 lg:grid-cols-2">
      <ProjectCard
        title="VoiceGraph"
        desc="Voice-first interactive knowledge graph with Gemini Live audio, Three.js graph rendering, and GraphRAG-style navigation over 569 nodes and 882 edges."
        tags={['Gemini Live', 'LangChain', 'Neo4j', 'Three.js', 'FastAPI']}
        color={accents.purple}
        link="https://voicegraph-802587268683.us-central1.run.app/"
      />
      <ProjectCard
        title="GyBuddy"
        desc="AI running companion on React Native that uses Gemini 2.5 Flash Native Audio for real-time coaching, tools, route awareness, and treadmill detection."
        tags={['React Native', 'Expo', 'Gemini Live', 'Maps API', 'Firebase']}
        color={accents.amber}
        link="https://github.com/bhaveshgupta01"
      />
      <ProjectCard
        title="SignalFlow"
        desc="Autonomous crypto trading agent coordinating asynchronous market signals, wallet activity, and Gemini-based evaluation over a large tool surface."
        tags={['Agentic AI', 'Gemini Vertex', 'MCP', 'Hyperliquid', 'Streamlit']}
        color={accents.rose}
        link="https://github.com/bhaveshgupta01"
      />
      <ProjectCard
        title="Virtual War Room"
        desc="Top Technical Build at Google × Columbia: a multi-agent voice boardroom with four AI personas, raise-hand interjections, and structured voting."
        tags={['Google ADK', 'Gemini Live', 'FastAPI', 'React', 'WebSocket']}
        color={accents.blue}
        link="https://github.com/bhaveshgupta01"
      />
      <ProjectCard
        title="Pulse NYC"
        desc="1st place AI marketing prototype with event-triggered campaign deployment, real-time audience targeting, and rapid validation loops."
        tags={['Next.js', 'TypeScript', 'AI', 'Real-time']}
        color={accents.emerald}
        link="https://github.com/bhaveshgupta01"
      />
      <ProjectCard
        title="Agentic RAG Engine"
        desc="Self-correcting RAG engine that grades retrieval quality and falls back to web search when local knowledge is weak. This powers the current portfolio agent."
        tags={['LangGraph', 'ChromaDB', 'Gemini', 'RAG']}
        color={accents.purple}
        link="https://github.com/bhaveshgupta01"
      />
    </div>
  </div>
);

const LeadershipContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Award} color={accents.rose} title="Leadership & Recognition" kicker="Leadership">
      The AI can send users here for signal beyond projects: facilitation, community building, teaching, and team ownership.
    </SectionHeading>

    <div className="space-y-4">
      <WorkCard
        role="Android Lead & Google Facilitator"
        company="Google Developer Student Club — GGSIPU"
        date="2022 – 2023"
        color={accents.emerald}
        desc="Taught 50+ students in Jetpack Compose through 12+ workshops and iterated the curriculum week over week."
      />
      <WorkCard
        role="Technical Festival Lead — Infoxpression"
        company="GGSIPU"
        date="2022 – 2024"
        color={accents.amber}
        desc="Led 30+ events across a 3-day technical festival, coordinating both operations and technical execution."
      />
      <WorkCard
        role="Hackathon Wins"
        company="NYC"
        date="2025 – 2026"
        color={accents.blue}
        desc="1st Place at Pulse NYC and Top Technical Build at Google × Columbia for Virtual War Room."
      />
    </div>
  </div>
);

const PublicationsContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={BookOpen} color={accents.purple} title="Publications" kicker="Research">
      Research still matters in the wallet version, but it gets surfaced as one dense, high-signal card instead of a standalone page section.
    </SectionHeading>

    <a
      href="https://doi.org/10.24874/PES.SI.25.03A.007"
      target="_blank"
      rel="noopener noreferrer"
      className="group block rounded-[28px] border border-subtle/15 bg-surface-2/70 p-6 transition-colors hover:border-violet-500/30"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-lg font-bold italic tracking-tight text-ink transition-colors group-hover:text-violet-400">
            &quot;Comparative Analysis of Deep Transfer Learning Techniques for Mammographic Image Classification&quot;
          </p>
          <p className="mt-3 text-sm text-muted">
            Proceedings on Engineering Sciences · ICAIA 2024 ·
            {' '}
            <span className="text-violet-500 dark:text-violet-400">Best Presenter Award</span>
          </p>
        </div>
        <ExternalLink size={18} className="shrink-0 text-faint transition-colors group-hover:text-violet-400" />
      </div>
      <div className="mt-5 flex flex-wrap gap-2">
        {['Transfer Learning', 'ResNet50', 'Medical Imaging', 'DOI-indexed'].map((item) => (
          <span key={item} className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${accents.purple}`}>
            {item}
          </span>
        ))}
      </div>
    </a>
  </div>
);

const DesignContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Palette} color={accents.amber} title="Design Gallery" kicker="Visual Work">
      Instead of sending you to a separate route immediately, the wallet card gives a condensed gallery pass with counts, preview assets, and a jump-off to the full design page.
    </SectionHeading>

    <div className="flex flex-wrap gap-2">
      {designCounts.map(({ category, count }) => (
        <span key={category} className="rounded-full border border-subtle/15 bg-surface-2/70 px-3 py-1.5 text-xs font-medium text-ink">
          {category}
          {' '}
          <span className="text-faint">({count})</span>
        </span>
      ))}
    </div>

    <div className="grid gap-4 lg:grid-cols-2">
      {featuredDesigns.map((design) => (
        <div key={design.id} className="overflow-hidden rounded-[24px] border border-subtle/15 bg-surface-2/70">
          <div className="aspect-[16/10] overflow-hidden bg-black/20">
            <img src={design.image} alt={design.title} className="h-full w-full object-cover" loading="lazy" />
          </div>
          <div className="p-4">
            <div className="text-sm font-bold tracking-tight text-ink">{design.title}</div>
            <div className="mt-1 text-xs uppercase tracking-[0.18em] text-faint">{design.category}</div>
            <p className="mt-3 text-[13px] leading-relaxed text-muted">{design.description}</p>
          </div>
        </div>
      ))}
    </div>

    <div className="flex flex-wrap gap-3">
      <Link
        to="/design"
        className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2.5 text-sm font-semibold text-app transition-opacity hover:opacity-90"
      >
        <Palette size={15} />
        Open full design gallery
      </Link>
      <span className="inline-flex items-center gap-2 rounded-full border border-subtle/15 px-4 py-2.5 text-sm text-muted">
        Canva, Photoshop, Illustrator, CorelDraw, Figma
      </span>
    </div>
  </div>
);

const ContactContent = () => (
  <div className="space-y-6">
    <SectionHeading icon={Mail} color={accents.rose} title="Contact" kicker="Reach Out">
      The contact card stays tactile. It still lets someone save the vCard immediately, but now it behaves like the last card in a wallet stack.
    </SectionHeading>

    <button
      type="button"
      onClick={downloadVCard}
      className="group relative w-full overflow-hidden rounded-[28px] border border-subtle/15 bg-surface-2/70 p-6 text-left transition-colors hover:border-accent/25"
    >
      <LogoIcon
        aria-hidden
        className="pointer-events-none absolute -bottom-6 -right-6 h-40 w-40 text-ink opacity-10"
        strokeWidth={30}
      />
      <div className="relative z-10 flex items-start gap-4">
        <div className="h-20 w-20 overflow-hidden rounded-2xl bg-surface ring-1 ring-subtle/15">
          <img src="/profile_photo.png" alt="Bhavesh Gupta" className="h-full w-full object-cover" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h4 className="text-2xl font-bold tracking-tight text-ink">Bhavesh Gupta</h4>
            <span className="rounded-full bg-accent/12 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-accent">
              vCard
            </span>
          </div>
          <p className="mt-2 text-sm text-muted">MSCS @ NYU Courant · Graduate Assistant @ NYUAD NY</p>
          <p className="mt-2 text-sm leading-relaxed text-faint">Building AI for problems that matter.</p>
        </div>
        <div className="hidden h-16 w-16 shrink-0 rounded-xl bg-white p-1 md:block">
          <img src="/qrcode1.jpeg" alt="QR code" className="h-full w-full object-contain" />
        </div>
      </div>

      <div className="relative z-10 mt-6 grid gap-3 sm:grid-cols-2">
        <ContactRow icon={Mail} label="Email" value="bg2896@nyu.edu" />
        <ContactRow icon={Phone} label="Phone" value="+1 (201) 492-8876" />
        <ContactRow icon={MapPin} label="Location" value="Manhattan, NY" />
        <ContactRow icon={QrCode} label="Web" value="libralpanda.vercel.app" />
      </div>

      <div className="relative z-10 mt-6 flex flex-wrap items-center justify-between gap-4">
        <div className="text-xs text-faint">
          Adds directly to Contacts on iOS, Android, macOS, Gmail, and Outlook.
        </div>
        <span className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2.5 text-sm font-semibold text-app">
          <Download size={15} />
          Save Contact
        </span>
      </div>
    </button>

    <div className="flex flex-wrap gap-3">
      <a href="mailto:bg2896@nyu.edu" className="glass-chip inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm text-ink hover:text-accent">
        <Mail size={14} />
        Email
      </a>
      <a href="https://linkedin.com/in/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="glass-chip inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm text-ink hover:text-accent">
        <Linkedin size={14} />
        LinkedIn
      </a>
      <a href="https://github.com/bhaveshgupta01" target="_blank" rel="noopener noreferrer" className="glass-chip inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm text-ink hover:text-accent">
        <Github size={14} />
        GitHub
      </a>
    </div>
  </div>
);

export const WALLET_CARD_DEFS = [
  {
    id: 'about',
    title: 'About Me',
    eyebrow: 'Identity',
    summary: 'Who I am, what I build, and the lens I use to choose projects.',
    highlights: ['NYU Courant', 'Applied AI', 'Product-minded engineering'],
    gradient: 'from-rose-500/18 via-rose-400/4 to-transparent',
    icon: User,
    render: AboutContent,
  },
  {
    id: 'journey',
    title: 'Journey',
    eyebrow: 'Trajectory',
    summary: 'Android to ML to domain-specific AI, with each stage changing the shape of the next.',
    highlights: ['Android roots', 'Healthcare AI', 'Agentic systems'],
    gradient: 'from-amber-400/18 via-amber-300/4 to-transparent',
    icon: Map,
    render: JourneyContent,
  },
  {
    id: 'experience',
    title: 'Experience',
    eyebrow: 'Work',
    summary: 'Enterprise systems, research internships, and developer education.',
    highlights: ['TCS', 'Synopsys', 'NYUAD'],
    gradient: 'from-blue-500/18 via-blue-400/4 to-transparent',
    icon: Briefcase,
    render: ExperienceContent,
  },
  {
    id: 'skills',
    title: 'Skills',
    eyebrow: 'Stack',
    summary: 'The language, AI, mobile, backend, and infrastructure stack behind the work.',
    highlights: ['LangGraph', 'FastAPI', 'React 19'],
    gradient: 'from-amber-500/16 via-yellow-300/4 to-transparent',
    icon: Cpu,
    render: SkillsContent,
  },
  {
    id: 'projects',
    title: 'Projects',
    eyebrow: 'Builds',
    summary: 'The portfolio’s highest-signal work, from VoiceGraph to multi-agent systems.',
    highlights: ['VoiceGraph', 'GyBuddy', 'SignalFlow'],
    gradient: 'from-violet-500/18 via-blue-400/4 to-transparent',
    icon: Folder,
    render: ProjectsContent,
  },
  {
    id: 'leadership',
    title: 'Leadership',
    eyebrow: 'Community',
    summary: 'Work that shows teaching, facilitation, and responsibility beyond code shipping.',
    highlights: ['GDSC', 'Compose workshops', 'Hackathon wins'],
    gradient: 'from-rose-500/16 via-orange-300/4 to-transparent',
    icon: Award,
    render: LeadershipContent,
  },
  {
    id: 'publications',
    title: 'Publications',
    eyebrow: 'Research',
    summary: 'A compact research card with the mammography paper and award context.',
    highlights: ['ICAIA 2024', 'Best Presenter', 'Medical imaging'],
    gradient: 'from-violet-500/18 via-violet-300/4 to-transparent',
    icon: BookOpen,
    render: PublicationsContent,
  },
  {
    id: 'design',
    title: 'Design',
    eyebrow: 'Gallery',
    summary: 'A wallet-native preview of the visual design archive, with a link to the full route.',
    highlights: ['Posters', 'Banners', 'Social assets'],
    gradient: 'from-amber-500/18 via-rose-300/4 to-transparent',
    icon: Palette,
    render: DesignContent,
  },
  {
    id: 'contact',
    title: 'Contact',
    eyebrow: 'Reach Out',
    summary: 'The last card in the stack. Save the vCard or jump out through email, LinkedIn, or GitHub.',
    highlights: ['vCard', 'NYC', 'Summer 2026 roles'],
    gradient: 'from-emerald-500/18 via-blue-300/4 to-transparent',
    icon: Mail,
    render: ContactContent,
  },
];
