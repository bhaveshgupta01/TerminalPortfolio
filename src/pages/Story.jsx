import { useEffect, useRef, useState } from 'react';
import Lines, { Mark } from '../components/story/Lines';
import ChatDock from '../components/story/ChatDock';
import { useTheme } from '../context/ThemeContext';
import { designs } from '../data/designs';
import { DESIGN_SIZES } from '../data/designSizes';
import { downloadVCard } from '../lib/vcard';
import {
  PERSON, ERAS, CHAPTERS, INTERNSHIPS, TCS, PROJECTS, EXTRAS, HONORS, SECTIONS,
} from '../data/story';
import './story.css';

const Dot = ({ era }) => <i className={`st-dot st-${era}`} aria-hidden="true" />;
const pad = (n) => String(n).padStart(2, '0');
const LABEL = Object.fromEntries(SECTIONS.map((s) => [s.id, s.label]));

const FILTERS = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'nyu', label: 'NYU', test: (p) => p.era === 'nyu' },
  { id: 'ipu', label: 'IPU', test: (p) => p.era === 'ipu' },
  { id: 'wins', label: 'Award winners', test: (p) => !!p.award },
];

// Collage: every design keeps its own proportions. Each tile spans a number
// of rows (out of 6) chosen from its shape, and as many columns as that
// height needs; the grid packs them densely and scrolls sideways.
const small = (src) => src.replace('/design/', '/design/thumbs/').replace(/\.png$/, '.jpg');
const large = (src) => src.replace('/design/', '/design/large/').replace(/\.png$/, '.jpg');
const COLLAGE = [...designs]
  .sort((a, b) => ((a.id * 7) % 19) - ((b.id * 7) % 19)) // fixed shuffle so categories mix
  .map((d, i) => {
    const [w, h] = DESIGN_SIZES[d.image] || [1, 1];
    const ratio = w / h;
    let rows;
    if (ratio >= 1.7) rows = i % 2 ? 2 : 3;
    else if (ratio > 1.15) rows = 3;
    else if (ratio > 0.85) rows = i % 2 ? 3 : 2;
    else if (ratio > 0.6) rows = i % 3 === 0 ? 6 : 4;
    else rows = 6;
    return { ...d, rows, cols: Math.max(1, Math.round(rows * ratio)) };
  });

export default function Story() {
  const rootRef = useRef(null);
  const carouselRef = useRef(null);
  const focusRef = useRef(null); // which line is emphasised right now
  const { theme, toggle } = useTheme();
  const [sectionIdx, setSectionIdx] = useState(0);
  const [activeProject, setActiveProject] = useState(PROJECTS[0].id);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [lightbox, setLightbox] = useState(null);
  const [filter, setFilter] = useState('all');
  const [copied, setCopied] = useState(false);
  const drag = useRef({ down: false, moved: false, x: 0, left: 0 });

  // Opening sequence: only when the visitor lands at the top of the page.
  const [intro, setIntro] = useState(() => (
    typeof window !== 'undefined'
    && !window.location.hash
    && window.scrollY < 8
    && !window.matchMedia('(prefers-reduced-motion: reduce)').matches
  ));
  useEffect(() => {
    if (!intro) return undefined;
    const id = setTimeout(() => setIntro(false), 2600);
    return () => clearTimeout(id);
  }, [intro]);

  // This page owns the document scroll and background while it is mounted.
  useEffect(() => {
    document.documentElement.classList.add('story-page');
    return () => document.documentElement.classList.remove('story-page');
  }, []);

  // Deep links: /#projects lands on that section.
  useEffect(() => {
    const id = window.location.hash.slice(1);
    if (id) document.getElementById(id)?.scrollIntoView({ block: 'start' });
  }, []);

  // Reveal each section's content the first time it comes into view.
  useEffect(() => {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.3 });
    rootRef.current.querySelectorAll('.st-sec').forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, []);

  const goTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const openProject = (id) => {
    setActiveProject(id);
    if (window.matchMedia('(max-width: 899px)').matches) setSheetOpen(true);
  };

  // The chat can move the page.
  const onNavigate = (section, project) => {
    if (project) {
      goTo('projects');
      openProject(project);
    } else if (section) {
      goTo(section);
    }
  };

  useEffect(() => {
    if (!sheetOpen && !lightbox) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') { setSheetOpen(false); setLightbox(null); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sheetOpen, lightbox]);

  const slide = (dir) => {
    const el = carouselRef.current;
    if (el) el.scrollBy({ left: dir * el.clientWidth * 0.7, behavior: 'smooth' });
  };

  // Hovering anything that belongs to an era lights up that era's line.
  const era = (key) => ({
    onMouseEnter: () => { focusRef.current = key; },
    onMouseLeave: () => { focusRef.current = null; },
  });

  const shown = PROJECTS.filter(FILTERS.find((f) => f.id === filter).test);
  const project = PROJECTS.find((p) => p.id === activeProject) || PROJECTS[0];
  const shownIdx = shown.findIndex((p) => p.id === project.id);
  const step = (dir) => {
    const next = shown[(Math.max(0, shownIdx) + dir + shown.length) % shown.length];
    setActiveProject(next.id);
    focusRef.current = null;
  };
  const pickFilter = (id) => {
    setFilter(id);
    const list = PROJECTS.filter(FILTERS.find((f) => f.id === id).test);
    if (!list.some((p) => p.id === activeProject)) setActiveProject(list[0].id);
  };

  // Drag the collage sideways with the mouse; a drag is not a click.
  const dragHandlers = {
    onPointerDown: (e) => {
      if (e.pointerType !== 'mouse') return;
      drag.current = { down: true, moved: false, x: e.clientX, left: carouselRef.current.scrollLeft };
    },
    onPointerMove: (e) => {
      const d = drag.current;
      if (!d.down) return;
      const dx = e.clientX - d.x;
      if (Math.abs(dx) > 5) d.moved = true;
      if (d.moved) carouselRef.current.scrollLeft = d.left - dx;
    },
    onPointerUp: () => { drag.current.down = false; },
    onPointerLeave: () => { drag.current.down = false; },
  };

  const copyEmail = async () => {
    try {
      await navigator.clipboard.writeText(PERSON.email);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch { /* clipboard blocked: the mailto link still works */ }
  };

  return (
    <div className={`story${intro ? ' is-intro' : ''}`} ref={rootRef}>
      <Lines rootRef={rootRef} onSection={setSectionIdx} focusRef={focusRef} intro={intro} />

      <header className="st-nav">
        <button type="button" className="st-brand" onClick={() => goTo('hero')} aria-label="Back to top">
          <Mark size={26} />
          <span>{PERSON.name}</span>
        </button>
        <nav>
          <a href="/v1">v1</a>
          <a href="/v2">v2</a>
          <button type="button" onClick={toggle} aria-label="Switch theme">
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </nav>
      </header>

      <div className="st-legend">
        {Object.entries(ERAS).map(([k, e]) => (
          <span key={k} {...era(k)}><Dot era={k} />{e.label}<em>{e.years}</em></span>
        ))}
      </div>

      <div className="st-rail" aria-hidden="true">
        <span>{pad(sectionIdx + 1)}</span>{SECTIONS[sectionIdx]?.label}
      </div>

      <nav className="st-ticks" aria-label="Sections">
        {SECTIONS.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className={i === sectionIdx ? 'is-active' : ''}
            onClick={() => goTo(s.id)}
            aria-label={s.label}
            aria-current={i === sectionIdx ? 'true' : undefined}
          >
            <span>{s.label}</span>
          </button>
        ))}
      </nav>

      <main>
        {/* 1 — Hi */}
        <section id="hero" data-pose="hero" className="st-sec">
          <div className="st-hero">
            <div className="st-mark-box" data-a="mark" />
            <h1 className="st-hi">Hi</h1>
          </div>
          <p className="st-hint">Scroll<i aria-hidden="true" /></p>
        </section>

        {/* 2 — Bhavesh */}
        <section id="name" data-pose="name" className="st-sec">
          <div className="st-name-wrap">
            <p className="st-kicker">I’m</p>
            <h2 className="st-name" data-a="name" aria-label="Bhavesh">
              {'Bhavesh'.split('').map((ch, i) => (
                <span key={i} style={{ '--i': i }} aria-hidden="true">{ch}</span>
              ))}
            </h2>
            <p className="st-lead">{PERSON.line}</p>
          </div>
        </section>

        {/* 3 — IPU / TCS / NYU */}
        <section id="chapters" data-pose="chapters" className="st-sec">
          <div className="st-chapters">
            {CHAPTERS.map((c) => (
              <article key={c.era} data-a={`ch-${c.era}`} className="st-chapter" {...era(c.era)}>
                <p className="st-mono">{ERAS[c.era].years}</p>
                <h3>{ERAS[c.era].label}</h3>
                <p className="st-strong">{c.title}</p>
                <p className="st-muted">{c.place}</p>
                <p>{c.text}</p>
              </article>
            ))}
          </div>
        </section>

        {/* 4 — Internships */}
        <section id="internships" data-pose="internships" className="st-sec">
          <div className="st-col">
            <h2 className="st-h">{LABEL.internships}</h2>
            <div className="st-items">
              {INTERNSHIPS.map((i) => (
                <article key={i.id} data-a={i.id} className="st-item" {...era(i.era)}>
                  <p className="st-mono"><Dot era={i.era} />{i.when}</p>
                  <h3>{i.company}</h3>
                  <p className="st-strong">{i.role}</p>
                  <p>{i.text}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* 5 — TCS */}
        <section id="tcs" data-pose="tcs" className="st-sec">
          <div className="st-col">
            <h2 className="st-h">{LABEL.tcs}</h2>
            <p className="st-mono"><Dot era="tcs" />{TCS.when}</p>
            <h3 className="st-big">{TCS.company}</h3>
            <p className="st-strong">{TCS.role} · {TCS.client}</p>
            <dl className="st-stats">
              {TCS.stats.map((s) => (
                <div key={s.label}><dt><CountUp value={s.value} /></dt><dd>{s.label}</dd></div>
              ))}
            </dl>
            <ul className="st-points">
              {TCS.points.map((p) => <li key={p}>{p}</li>)}
            </ul>
          </div>
        </section>

        {/* 6 — Projects */}
        <section id="projects" data-pose="projects" className="st-sec">
          <div className="st-projects" data-a="projects">
            <div className="st-pleft">
              <h2 className="st-h">{LABEL.projects} <sup>{PROJECTS.length}</sup></h2>
              <div className="st-filters" role="tablist" aria-label="Filter projects">
                {FILTERS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    role="tab"
                    aria-selected={filter === f.id}
                    className={filter === f.id ? 'is-active' : ''}
                    onClick={() => pickFilter(f.id)}
                  >
                    {f.label}<span>{PROJECTS.filter(f.test).length}</span>
                  </button>
                ))}
              </div>
              <ol className="st-plist" onMouseLeave={() => { focusRef.current = null; }}>
                {shown.map((p, i) => (
                  <li key={p.id}>
                    <button
                      type="button"
                      className={`st-${p.era}-row${p.id === activeProject ? ' is-active' : ''}`}
                      onMouseEnter={() => { setActiveProject(p.id); focusRef.current = p.era; }}
                      onFocus={() => setActiveProject(p.id)}
                      onClick={() => openProject(p.id)}
                    >
                      <span className="st-pnum">{pad(i + 1)}</span>
                      <span className="st-pname">{p.name}</span>
                      {p.award && <span className="st-pbadge">{p.award.split(',')[0]}</span>}
                    </button>
                  </li>
                ))}
              </ol>
            </div>
            <aside className="st-pdetail" aria-live="polite">
              <ProjectDetail project={project} index={shownIdx} total={shown.length} onStep={step} />
            </aside>
          </div>
        </section>

        {/* 7 — Extras */}
        <section id="extras" data-pose="extras" className="st-sec">
          <div className="st-col">
            <h2 className="st-h">{LABEL.extras}</h2>
            <div className="st-items">
              <div data-a="ex-ipu" className="st-item" {...era('ipu')}>
                <p className="st-mono"><Dot era="ipu" />Leadership · bachelor’s</p>
                <ul className="st-rows">
                  {EXTRAS.ipu.map((e) => (
                    <li key={e.title}><strong>{e.title}</strong><span className="st-muted">{e.org}</span><span>{e.text}</span></li>
                  ))}
                </ul>
              </div>
              <div data-a="ex-nyu" className="st-item" {...era('nyu')}>
                <p className="st-mono"><Dot era="nyu" />Volunteering · NYU</p>
                <ul className="st-rows">
                  {EXTRAS.nyu.map((e) => (
                    <li key={e.title}><strong>{e.title}</strong><span className="st-muted">{e.org}</span><span>{e.text}</span></li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 8 — Research / Scholarship / Hackathon wins */}
        <section id="honors" data-pose="honors" className="st-sec">
          <div className="st-honors">
            <article data-a="h-research" {...era('ipu')}>
              <p className="st-mono"><Dot era="ipu" />Research</p>
              <h3>{HONORS.research.title}</h3>
              <p className="st-muted">{HONORS.research.meta}</p>
              <p><a href={HONORS.research.link} target="_blank" rel="noopener noreferrer">Read the paper ↗</a></p>
            </article>
            <article data-a="h-scholar" {...era('nyu')}>
              <p className="st-mono"><Dot era="nyu" />Scholarship</p>
              <h3>{HONORS.scholarship.title}</h3>
              <p className="st-muted">{HONORS.scholarship.meta}</p>
            </article>
            <article data-a="h-hack" {...era('nyu')}>
              <p className="st-mono"><Dot era="nyu" />Hackathon wins</p>
              <ul className="st-wins">
                {HONORS.hackathons.map((h) => (
                  <li key={h.name}>
                    <button type="button" onClick={() => { goTo('projects'); openProject(h.project); }}>
                      <strong>{h.place}</strong>
                      <span>{h.name}{h.when ? ` · ${h.when}` : ''}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </article>
          </div>
        </section>

        {/* 9 — Gallery */}
        <section id="gallery" data-pose="gallery" className="st-sec st-sec-wide">
          <div className="st-gallery-head">
            <h2 className="st-h">{LABEL.gallery}</h2>
            <p className="st-muted">Posters, banners and invitations I designed for university events. Drag to look around. <a href="/design">See them all ↗</a></p>
            <div className="st-arrows">
              <button type="button" onClick={() => slide(-1)} aria-label="Scroll designs left">←</button>
              <button type="button" onClick={() => slide(1)} aria-label="Scroll designs right">→</button>
            </div>
          </div>
          <div className="st-collage" data-a="carousel" ref={carouselRef} {...dragHandlers}>
            {COLLAGE.map((d) => (
              <button
                key={d.id}
                type="button"
                style={{ gridRow: `span ${d.rows}`, gridColumn: `span ${d.cols}` }}
                onClick={() => { if (!drag.current.moved) setLightbox(d); }}
                aria-label={`Open ${d.title}`}
              >
                <img src={small(d.image)} alt={d.title} loading="lazy" draggable="false" />
                <span>{d.title}</span>
              </button>
            ))}
          </div>
        </section>

        {/* 10 — Contact */}
        <section id="contact" data-pose="contact" className="st-sec">
          <div className="st-contact">
            <div className="st-contact-text">
              <h2 className="st-big">Say hi.</h2>
              <p className="st-lead-2">Have a problem worth solving, a role that fits, or a hackathon team that needs one more? I would like to hear about it.</p>
              <ul className="st-ways">
                <li>
                  <span className="st-mono">Email</span>
                  <a href={`mailto:${PERSON.email}`}>{PERSON.email}</a>
                  <button type="button" onClick={copyEmail}>{copied ? 'Copied' : 'Copy'}</button>
                </li>
                <li>
                  <span className="st-mono">LinkedIn</span>
                  <a href={PERSON.linkedin} target="_blank" rel="noopener noreferrer">in/bhaveshgupta01 ↗</a>
                </li>
                <li>
                  <span className="st-mono">GitHub</span>
                  <a href={PERSON.github} target="_blank" rel="noopener noreferrer">bhaveshgupta01 ↗</a>
                </li>
                <li>
                  <span className="st-mono">Based in</span>
                  <span>{PERSON.location}</span>
                </li>
              </ul>
              <p><button type="button" className="st-cta" onClick={downloadVCard}>Save my contact card</button></p>
            </div>
            <div className="st-mark-box st-mark-end" data-a="mark-end" />
          </div>
          <div className="st-panda" aria-hidden="true">
            <div className="st-panda-body">
              <svg className="st-panda-paw" viewBox="0 0 10 16" shapeRendering="crispEdges">
                <rect x="2" y="7" width="6" height="9" fill="#9e9e9e" />
                <rect x="3" y="7" width="4" height="9" fill="#f4f4f4" />
                <rect x="3" y="0" width="4" height="1" fill="#1c1c1c" />
                <rect x="2" y="1" width="6" height="6" fill="#1c1c1c" />
                <rect x="1" y="2" width="8" height="4" fill="#1c1c1c" />
              </svg>
              <img src="/panda-formal.png" alt="" />
            </div>
          </div>
        </section>
      </main>

      <ChatDock onNavigate={onNavigate} />

      {sheetOpen && (
        <div className="st-sheet" role="dialog" aria-modal="true" aria-label={project.name} onClick={() => setSheetOpen(false)}>
          <div className="st-sheet-card" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="st-sheet-close" onClick={() => setSheetOpen(false)}>Close</button>
            <ProjectDetail project={project} />
          </div>
        </div>
      )}

      {lightbox && (
        <div className="st-sheet st-lightbox" role="dialog" aria-modal="true" aria-label={lightbox.title} onClick={() => setLightbox(null)}>
          <figure onClick={(e) => e.stopPropagation()}>
            <img src={large(lightbox.image)} alt={lightbox.title} />
            <figcaption>
              <strong>{lightbox.title}</strong>
              <span>{lightbox.description}</span>
              <button type="button" onClick={() => setLightbox(null)}>Close</button>
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}

function ProjectDetail({ project: p, index, total, onStep }) {
  return (
    <div className={`st-pd st-pd-${p.era}`} key={p.id}>
      <div className="st-pd-top st-mono">
        <span><Dot era={p.era} />{ERAS[p.era].label} · {p.context}</span>
        {total > 0 && index >= 0 && <span>{pad(index + 1)} / {pad(total)}</span>}
      </div>
      <h3>{p.name}</h3>
      <p className="st-strong">{p.tagline}</p>
      {p.award && <p className="st-award">{p.award}</p>}
      <p>{p.text}</p>
      {p.stats && (
        <dl className="st-pd-stats">
          {p.stats.map(([value, label]) => (
            <div key={label}><dt>{value}</dt><dd>{label}</dd></div>
          ))}
        </dl>
      )}
      <ul className="st-chips">
        {p.stack.map((t) => <li key={t}>{t}</li>)}
      </ul>
      <div className="st-pd-foot">
        {p.repo
          ? <a className="st-cta" href={p.repo} target="_blank" rel="noopener noreferrer">{p.repoLabel || 'View on GitHub'} ↗</a>
          : <span className="st-mono">Built at the event; no public repo</span>}
        {onStep && total > 1 && (
          <span className="st-pd-nav">
            <button type="button" onClick={() => onStep(-1)} aria-label="Previous project">←</button>
            <button type="button" onClick={() => onStep(1)} aria-label="Next project">→</button>
          </span>
        )}
      </div>
    </div>
  );
}

// "26M+" counts up from 0 the first time it scrolls into view.
function CountUp({ value }) {
  const ref = useRef(null);
  const match = /^([\d,.]+)(.*)$/.exec(value);
  useEffect(() => {
    const el = ref.current;
    if (!el || !match || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined;
    const target = parseFloat(match[1].replace(/,/g, ''));
    const decimals = (match[1].split('.')[1] || '').length;
    const format = (n) => n.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + match[2];
    let raf = 0;
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      const start = performance.now();
      const tick = (now) => {
        const u = Math.min(1, (now - start) / 1100);
        el.textContent = format(target * (1 - (1 - u) ** 3));
        if (u < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    }, { threshold: 0.6 });
    io.observe(el);
    return () => { io.disconnect(); cancelAnimationFrame(raf); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return <span ref={ref}>{value}</span>;
}
