import {
  User,
  Map,
  Briefcase,
  Cpu,
  Folder,
  Award,
  BookOpen,
  Palette,
  Mail,
} from 'lucide-react';
import AboutCard from './cards/AboutCard';
import JourneyCard from './cards/JourneyCard';
import ExperienceCard from './cards/ExperienceCard';
import SkillsCard from './cards/SkillsCard';
import ProjectsCard from './cards/ProjectsCard';
import LeadershipCard from './cards/LeadershipCard';
import PublicationsCard from './cards/PublicationsCard';
import DesignCard from './cards/DesignCard';
import ContactCard from './cards/ContactCard';

/**
 * Single source of truth for lab cards.
 *
 * `subsections` is used by:
 *   - the command palette to list jumpable anchors
 *   - the ui_command receiver to validate a `sectionId` from the AI
 *
 * `searchText` feeds the palette's fuzzy match.
 */
export const LAB_CARDS = [
  {
    id: 'about',
    title: 'About Me',
    eyebrow: 'Identity',
    summary: 'Who I am, what I build, and the lens I use to choose projects.',
    highlights: ['NYU Courant', 'Applied AI', 'Product-minded engineering'],
    gradient: 'from-rose-500/18 via-rose-400/4 to-transparent',
    icon: User,
    render: AboutCard,
    searchText: 'about identity who master nyu new york',
    subsections: [
      { id: 'identity', label: 'Identity' },
      { id: 'focus', label: 'Current Focus' },
      { id: 'values', label: 'Values' },
    ],
  },
  {
    id: 'journey',
    title: 'Journey',
    eyebrow: 'Trajectory',
    summary: 'Android → ML → agentic AI, with each stage changing the shape of the next.',
    highlights: ['Android roots', 'Healthcare AI', 'Agentic systems'],
    gradient: 'from-amber-400/18 via-amber-300/4 to-transparent',
    icon: Map,
    render: JourneyCard,
    searchText: 'journey trajectory path timeline android ml agentic',
    subsections: [
      { id: 'android', label: 'Era I — Android' },
      { id: 'ml', label: 'Era II — ML' },
      { id: 'agentic', label: 'Era III — Agentic AI' },
    ],
  },
  {
    id: 'experience',
    title: 'Experience',
    eyebrow: 'Work',
    summary: 'Enterprise systems, research internships, and developer education.',
    highlights: ['TCS', 'Synopsys', 'NYUAD'],
    gradient: 'from-blue-500/18 via-blue-400/4 to-transparent',
    icon: Briefcase,
    render: ExperienceCard,
    searchText: 'experience work role employer tcs northern trust synopsys esec nyuad graduate assistant',
    subsections: [
      { id: 'nyuad', label: 'NYU Abu Dhabi' },
      { id: 'tcs', label: 'Tata Consultancy · Northern Trust' },
      { id: 'synopsys', label: 'Synopsys' },
      { id: 'esec', label: 'eSec Forte' },
    ],
  },
  {
    id: 'skills',
    title: 'Skills',
    eyebrow: 'Stack',
    summary: 'Languages, AI, mobile, backend, frontend, and infra I ship with.',
    highlights: ['LangGraph', 'FastAPI', 'React 19'],
    gradient: 'from-amber-500/16 via-yellow-300/4 to-transparent',
    icon: Cpu,
    render: SkillsCard,
    searchText: 'skills stack technology tech kotlin python react langgraph pytorch',
    subsections: [
      { id: 'languages', label: 'Languages' },
      { id: 'ai-ml', label: 'AI / ML' },
      { id: 'mobile', label: 'Mobile' },
      { id: 'backend', label: 'Backend' },
      { id: 'frontend', label: 'Frontend' },
      { id: 'infra', label: 'Infra / Tools' },
    ],
  },
  {
    id: 'projects',
    title: 'Projects',
    eyebrow: 'Builds',
    summary: 'VoiceGraph, GyBuddy, SignalFlow, and more — the highest-signal builds.',
    highlights: ['VoiceGraph', 'GyBuddy', 'SignalFlow'],
    gradient: 'from-violet-500/18 via-blue-400/4 to-transparent',
    icon: Folder,
    render: ProjectsCard,
    searchText: 'projects builds voicegraph gybuddy signalflow war room pulse nyc agentic rag',
    subsections: [
      { id: 'voicegraph', label: 'VoiceGraph' },
      { id: 'gybuddy', label: 'GyBuddy' },
      { id: 'signalflow', label: 'SignalFlow' },
      { id: 'virtual-war-room', label: 'Virtual War Room' },
      { id: 'pulse-nyc', label: 'Pulse NYC' },
      { id: 'agentic-rag', label: 'Agentic RAG Engine' },
    ],
  },
  {
    id: 'leadership',
    title: 'Leadership',
    eyebrow: 'Community',
    summary: 'Teaching, facilitation, and wins beyond writing code alone.',
    highlights: ['GDSC', 'Compose workshops', 'Hackathon wins'],
    gradient: 'from-rose-500/16 via-orange-300/4 to-transparent',
    icon: Award,
    render: LeadershipCard,
    searchText: 'leadership community gdsc google developer student teaching workshops hackathons',
    subsections: [
      { id: 'gdsc', label: 'GDSC — GGSIPU' },
      { id: 'infoxpression', label: 'Infoxpression Festival' },
      { id: 'hackathons', label: 'Hackathon Wins' },
    ],
  },
  {
    id: 'publications',
    title: 'Publications',
    eyebrow: 'Research',
    summary: 'Peer-reviewed mammography research + ICAIA 2024 Best Presenter.',
    highlights: ['ICAIA 2024', 'Best Presenter', 'Medical imaging'],
    gradient: 'from-violet-500/18 via-violet-300/4 to-transparent',
    icon: BookOpen,
    render: PublicationsCard,
    searchText: 'publications research paper icaia mammography transfer learning resnet medical imaging',
    subsections: [
      { id: 'icaia-mammography', label: 'ICAIA 2024 Mammography Paper' },
    ],
  },
  {
    id: 'design',
    title: 'Design',
    eyebrow: 'Gallery',
    summary: 'Posters, banners, and social assets with a jump to the full archive.',
    highlights: ['Posters', 'Banners', 'Social'],
    gradient: 'from-amber-500/18 via-rose-300/4 to-transparent',
    icon: Palette,
    render: DesignCard,
    searchText: 'design visual gallery poster banner social figma photoshop illustrator',
    subsections: [
      { id: 'categories', label: 'Categories' },
      { id: 'featured', label: 'Featured Work' },
    ],
  },
  {
    id: 'contact',
    title: 'Contact',
    eyebrow: 'Reach Out',
    summary: 'vCard, email, LinkedIn, GitHub — save the card in a tap.',
    highlights: ['vCard', 'NYC', 'Summer 2026 roles'],
    gradient: 'from-emerald-500/18 via-blue-300/4 to-transparent',
    icon: Mail,
    render: ContactCard,
    searchText: 'contact email phone reach out linkedin github vcard',
    subsections: [
      { id: 'vcard', label: 'vCard' },
      { id: 'channels', label: 'Channels' },
    ],
  },
];

export const LAB_CARD_BY_ID = Object.fromEntries(LAB_CARDS.map((c) => [c.id, c]));
