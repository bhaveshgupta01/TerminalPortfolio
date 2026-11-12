import AboutContent from './cards/AboutContent';
import NowContent from './cards/NowContent';
import JourneyContent from './cards/JourneyContent';
import ExperienceContent from './cards/ExperienceContent';
import SkillsContent from './cards/SkillsContent';
import ProjectsContent from './cards/ProjectsContent';
import LeadershipContent from './cards/LeadershipContent';
import PublicationsContent from './cards/PublicationsContent';
import DesignContent from './cards/DesignContent';
import ContactContent from './cards/ContactContent';

/**
 * Wallet drawer cards.
 *
 * Each card has:
 *   id, label (8-10 char uppercase tag), title, subtitle,
 *   material (key from materials.js — drives the entire palette),
 *   keywords (kept for back-compat / palette typeahead),
 *   sections (subsection ids inside the card body — used by the agent's
 *             ui_command focus payload), render (the body component).
 *
 * Order matters — this is the order they sit in the drawer top-to-bottom.
 *
 * Recursive deck shape preserved from prior v2 so a portal card (e.g.
 * Projects → per-project sub-deck) can flip `kind: 'leaf' → 'portal'`
 * without restructuring callers.
 */

export const HOME_CARDS = [
  {
    id: 'about',
    kind: 'leaf',
    label: 'ABOUT',
    title: 'Bhavesh Gupta',
    subtitle: 'Building AI for problems that matter',
    material: 'graphite',
    keywords: ['who', 'about', 'identity', 'intro', 'hello'],
    sections: ['identity', 'focus', 'values'],
    render: AboutContent,
  },
  {
    id: 'now',
    kind: 'leaf',
    label: 'NOW',
    title: 'This week',
    subtitle: "What I'm building right now",
    material: 'emerald',
    keywords: ['now', 'current', 'this week', 'today', 'currently'],
    sections: ['updates'],
    render: NowContent,
  },
  {
    id: 'journey',
    kind: 'leaf',
    label: 'JOURNEY',
    title: 'The long road',
    subtitle: 'Android → ML → Agentic AI',
    material: 'bronze',
    keywords: ['journey', 'story', 'path', 'timeline'],
    sections: ['android', 'ml', 'agentic'],
    render: JourneyContent,
  },
  {
    id: 'experience',
    kind: 'leaf',
    label: 'EXPERIENCE',
    title: 'Work',
    subtitle: 'NYUAD · TCS · Synopsys · eSec Forte',
    material: 'cobalt',
    keywords: ['experience', 'work', 'job', 'role', 'tcs', 'synopsys'],
    sections: ['nyuad', 'tcs', 'synopsys', 'esec'],
    render: ExperienceContent,
  },
  {
    id: 'skills',
    kind: 'leaf',
    label: 'SKILLS',
    title: 'Stack',
    subtitle: 'Languages · AI · Mobile · Backend · Infra',
    material: 'titanium',
    keywords: ['skills', 'stack', 'tech', 'languages'],
    sections: ['languages', 'ai-ml', 'mobile', 'backend', 'frontend', 'infra'],
    render: SkillsContent,
  },
  {
    id: 'projects',
    kind: 'leaf', // flip to 'portal' when sub-deck cards land
    label: 'PROJECTS',
    title: 'Things I built',
    subtitle: 'VoiceGraph · GyBuddy · SignalFlow',
    material: 'copper',
    keywords: ['projects', 'built', 'made', 'voicegraph', 'gybuddy'],
    sections: ['voicegraph', 'gybuddy', 'signalflow', 'virtual-war-room', 'pulse-nyc', 'agentic-rag'],
    render: ProjectsContent,
  },
  {
    id: 'design',
    kind: 'leaf',
    label: 'DESIGN',
    title: 'Visual work',
    subtitle: 'Posters · Banners · Brand · Social',
    material: 'amethyst',
    keywords: ['design', 'visual', 'poster', 'figma'],
    sections: ['categories', 'featured'],
    render: DesignContent,
  },
  {
    id: 'publications',
    kind: 'leaf',
    label: 'PUBLICATIONS',
    title: 'Research',
    subtitle: 'ICAIA 2024 · Best Presenter Award',
    material: 'brass',
    keywords: ['publications', 'paper', 'research', 'icaia', 'mammography'],
    sections: ['icaia-mammography'],
    render: PublicationsContent,
  },
  {
    id: 'leadership',
    kind: 'leaf',
    label: 'LEADERSHIP',
    title: 'Beyond code',
    subtitle: 'GDSC · Festivals · Hackathon wins',
    material: 'ruby',
    keywords: ['leadership', 'community', 'gdsc', 'mentor'],
    sections: ['gdsc', 'infoxpression', 'hackathons'],
    render: LeadershipContent,
  },
  {
    id: 'contact',
    kind: 'leaf',
    label: 'CONTACT',
    title: 'Say hi',
    subtitle: 'Save the vCard · Manhattan · Open to work',
    material: 'gold',
    keywords: ['contact', 'email', 'reach', 'hire', 'vcard'],
    sections: ['vcard', 'channels'],
    render: ContactContent,
  },
];

export const DECKS = {
  home: {
    id: 'home',
    label: 'Home',
    parentDeck: null,
    cards: HOME_CARDS,
  },
};

export const getDeck = (deckId) => DECKS[deckId] || DECKS.home;

export const getCardInDeck = (deckId, cardId) => {
  const deck = getDeck(deckId);
  return deck.cards.find((c) => c.id === cardId) || null;
};

export const resolveStackRoute = (pathname) => {
  const trimmed = (pathname || '').replace(/^\/+|\/+$/g, '');
  if (!trimmed) return { deckId: 'home', cardId: null, open: false };
  const parts = trimmed.split('/');
  const maybeDeckId = parts[0];
  if (DECKS[maybeDeckId] && maybeDeckId !== 'home') {
    const sub = parts.slice(1);
    return { deckId: maybeDeckId, cardId: sub[0] || null, open: sub[1] === 'open' };
  }
  return { deckId: 'home', cardId: parts[0] || null, open: parts[1] === 'open' };
};

export const buildStackHref = ({ deckId = 'home', cardId = null, open = false } = {}) => {
  const segs = ['/v2'];
  if (deckId && deckId !== 'home') segs.push(deckId);
  if (cardId) segs.push(cardId);
  if (open) segs.push('open');
  return segs.join('/');
};
