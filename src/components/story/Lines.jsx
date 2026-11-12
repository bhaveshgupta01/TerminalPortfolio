import { useEffect, useRef } from 'react';

/**
 * The three lines.
 *
 * One fixed SVG sits behind the page. Every <section data-pose="..."> defines a
 * "pose": where each of the three lines should be when that section is on
 * screen. Poses are measured from real DOM boxes (elements tagged data-a="..."),
 * so the lines wrap the actual content at any screen size.
 *
 * On scroll we blend the pose of the section we are leaving into the pose of
 * the one we are entering. Each pose stays attached to its own section while
 * it scrolls, so the lines travel with the content and redraw in between.
 */

// Colours live in story.css so they can change with the theme.
const LINE_COLORS = {
  ipu: 'var(--ln-ipu)', // green  — bachelor's, GGSIPU
  tcs: 'var(--ln-tcs)', // orange — TCS
  nyu: 'var(--ln-nyu)', // violet — NYU
};
const KEYS = ['ipu', 'tcs', 'nyu'];
const N = 640; // points per line

/* ------------------------------ geometry ------------------------------ */

// Round every corner of a polyline with a fixed radius.
const fillet = (pts, R = 32) => {
  const out = [pts[0]];
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1];
    const [x, y] = pts[i];
    const [nx, ny] = pts[i + 1];
    const d1 = Math.hypot(x - px, y - py);
    const d2 = Math.hypot(nx - x, ny - y);
    const r = Math.min(R, d1 / 2, d2 / 2);
    if (r < 1) { out.push(pts[i]); continue; }
    const ax = x + ((px - x) / d1) * r;
    const ay = y + ((py - y) / d1) * r;
    const bx = x + ((nx - x) / d2) * r;
    const by = y + ((ny - y) / d2) * r;
    for (let k = 0; k <= 8; k++) {
      const u = k / 8;
      out.push([
        (1 - u) ** 2 * ax + 2 * u * (1 - u) * x + u * u * bx,
        (1 - u) ** 2 * ay + 2 * u * (1 - u) * y + u * u * by,
      ]);
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
};

// Resample a polyline to exactly n points, evenly spaced along its length.
const resample = (pts, n) => {
  const m = pts.length;
  const cum = new Float64Array(m);
  for (let i = 1; i < m; i++) {
    cum[i] = cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  }
  const total = cum[m - 1] || 1;
  const out = new Float32Array(n * 2);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const d = (total * k) / (n - 1);
    while (j < m - 2 && cum[j + 1] < d) j++;
    const seg = cum[j + 1] - cum[j] || 1;
    const u = Math.min(1, Math.max(0, (d - cum[j]) / seg));
    out[2 * k] = pts[j][0] + (pts[j + 1][0] - pts[j][0]) * u;
    out[2 * k + 1] = pts[j][1] + (pts[j + 1][1] - pts[j][1]) * u;
  }
  return out;
};

// Shift a polyline sideways by `dist`, keeping it parallel to the original
// (mitred at the corners), so strands drawn side by side never collapse onto
// each other.
const offsetLine = (pts, dist) => {
  if (!dist) return pts;
  const normals = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const dx = pts[i + 1][0] - pts[i][0];
    const dy = pts[i + 1][1] - pts[i][1];
    const len = Math.hypot(dx, dy) || 1;
    normals.push([-dy / len, dx / len]);
  }
  return pts.map(([x, y], i) => {
    const a = normals[Math.max(0, i - 1)];
    const b = normals[Math.min(normals.length - 1, i)];
    let mx = a[0] + b[0];
    let my = a[1] + b[1];
    const ml = Math.hypot(mx, my) || 1;
    mx /= ml;
    my /= ml;
    const k = dist / Math.max(0.3, mx * b[0] + my * b[1]);
    return [x + mx * k, y + my * k];
  });
};

const union = (a, b) => {
  const left = Math.min(a.left, b.left);
  const right = Math.max(a.right, b.right);
  const top = Math.min(a.top, b.top);
  const bottom = Math.max(a.bottom, b.bottom);
  return { left, right, top, bottom, w: right - left, h: bottom - top, cx: (left + right) / 2, cy: (top + bottom) / 2 };
};

/* ------------------------------- the mark ------------------------------ */

// Broken infinity with a twist. Six points, point-symmetric about the centre:
// two lobes left and right that never close, and two tall slashes through them.
const MARK_POINTS = [
  [-0.45, 0.9],
  [0.15, -0.9],
  [-0.95, -0.15],
  [0.95, 0.15],
  [-0.15, 0.9],
  [0.45, -0.9],
];

/* -------------------------------- poses -------------------------------- */

// Every pose runs from above the screen (TOP) to below it (BOT) so no line
// end is ever visible mid-transition.
const mk = (c) => {
  const TOP = -0.6 * c.h;
  const BOT = 1.6 * c.h;
  const thin = c.mobile ? 3.5 : 5;
  const clampX = (x) => Math.min(c.w - 6, Math.max(6, x));
  const radius = c.mobile ? 14 : 34;

  // A quiet line resting in the margin.
  const side = (which, k = 0) => {
    const inset = c.mobile ? 7 + k * 6 : c.w * 0.07 + k * 14;
    const x = which === 'L' ? inset : c.w - inset;
    return { pts: [[x, TOP], [x, BOT]], w: thin - 1.5, o: 0.5, soft: 1 };
  };

  // Serpentine around a box: over the top, down one side, under the bottom.
  const wrap = (r, pad, dir = 1) => {
    const a = clampX(dir > 0 ? r.right + pad : r.left - pad);
    const b = clampX(dir > 0 ? r.left - pad : r.right + pad);
    const pts = [
      [a, TOP],
      [a, r.top - pad],
      [b, r.top - pad],
      [b, r.bottom + pad],
      [a, r.bottom + pad],
      [a, BOT],
    ];
    return { pts: fillet(pts, radius), w: thin, o: 1, soft: 1 };
  };

  // Scribbled loops around a box.
  const loops = (r, k, turns, cw) => {
    const rx = Math.min(c.w / 2 - 8, r.w / 2 + 26 + k * 20);
    const ry = r.h / 2 + 6 + k * 13;
    const tilt = ((k - 1) * 5 * Math.PI) / 180;
    const dx = (k - 1) * (c.mobile ? 14 : 34);
    const pts = [[r.cx + dx, TOP]];
    const steps = Math.round(turns * 48);
    for (let i = 0; i <= steps; i++) {
      const a = -Math.PI / 2 + (cw ? 1 : -1) * (i / steps) * turns * 2 * Math.PI;
      // Each pass sits a little outside the last, like a pen circling a word,
      // so repeated turns read as separate strokes.
      const grow = 1 + 0.2 * (i / steps - 0.5);
      const x = Math.cos(a) * rx * grow;
      const y = Math.sin(a) * ry * grow;
      pts.push([
        r.cx + x * Math.cos(tilt) - y * Math.sin(tilt),
        r.cy + x * Math.sin(tilt) + y * Math.cos(tilt),
      ]);
    }
    // Leave by the margin so the lines clear the text below the name.
    const inset = c.mobile ? 7 + k * 6 : c.w * 0.135 + k * 14;
    const exitX = cw ? inset : c.w - inset;
    const exitY = r.bottom + 30 + k * 11;
    const tail = fillet([pts[pts.length - 1], [exitX, exitY], [exitX, BOT]], radius).slice(1);
    return { pts: [...pts, ...tail], w: thin, o: 1, soft: 1 };
  };

  // All three strands trace the mark, slightly out of register.
  const mark = (r, exit) => {
    const s = Math.min(r.w, r.h) / 2;
    const sw = Math.max(3.5, s * 0.055);
    const d = sw * 1.25;
    const base = MARK_POINTS.map(([x, y]) => [r.cx + x * s, r.cy + y * s]).reverse();
    let spine;
    if (exit) {
      // Opening: straight down from the top, through the mark, out the bottom.
      spine = [[base[0][0], TOP], ...base, [base[base.length - 1][0], BOT]];
    } else {
      // Closing: arrive from the right margin, run through the mark and carry
      // on off the bottom of the page.
      const x = c.w - (c.mobile ? 16 : c.w * 0.12);
      spine = [[x, TOP], [x, base[0][1]], ...base, [base[base.length - 1][0], BOT]];
    }
    const out = {};
    KEYS.forEach((key, i) => {
      out[key] = { pts: offsetLine(spine, (i - 1) * d), w: sw, o: 1, soft: 0.22 };
    });
    return out;
  };

  return { TOP, BOT, thin, side, wrap, loops, mark, clampX };
};

const POSES = {
  hero: (c) => mk(c).mark(c.rect('mark'), true),

  name: (c) => {
    const t = mk(c);
    const r = c.rect('name');
    return { ipu: t.loops(r, 2, 1.5, true), tcs: t.loops(r, 1, 1.5, false), nyu: t.loops(r, 0, 2.5, true) };
  },

  // Each chapter takes its colour: the line swells into a band behind its card.
  chapters: (c) => {
    const t = mk(c);
    const rs = { ipu: c.rect('ch-ipu'), tcs: c.rect('ch-tcs'), nyu: c.rect('ch-nyu') };
    const stacked = Math.abs(rs.ipu.left - rs.tcs.left) < 4;
    const out = {};
    KEYS.forEach((k) => {
      const r = rs[k];
      out[k] = stacked
        ? { pts: [[-c.w, r.cy], [2 * c.w, r.cy]], w: r.h + 18, o: 0.17 }
        : { pts: [[r.cx, t.TOP], [r.cx, t.BOT]], w: r.w + 22, o: 0.17 };
    });
    return out;
  },

  internships: (c) => {
    const t = mk(c);
    const p = c.mobile ? [8, 15] : [26, 46];
    return {
      ipu: t.wrap(union(c.rect('esec'), c.rect('synopsys')), p[0], 1),
      tcs: t.side('R', 2),
      nyu: t.wrap(c.rect('nyuad'), p[1], -1),
    };
  },

  // The whole page takes the TCS colour.
  tcs: (c) => {
    const t = mk(c);
    return {
      ipu: t.side('L', 0),
      tcs: { pts: [[c.w / 2, t.TOP], [c.w / 2, t.BOT]], w: c.w * 1.2, o: 0.15 },
      nyu: t.side('R', 0),
    };
  },

  projects: (c) => {
    const t = mk(c);
    return { ipu: t.side('L', 0), tcs: t.side('L', 1), nyu: t.wrap(c.rect('projects'), c.mobile ? 9 : 30, 1) };
  },

  extras: (c) => {
    const t = mk(c);
    const p = c.mobile ? [8, 15] : [26, 46];
    return {
      ipu: t.wrap(c.rect('ex-ipu'), p[0], -1),
      tcs: t.side('R', 2),
      nyu: t.wrap(c.rect('ex-nyu'), p[1], 1),
    };
  },

  // Research belongs to the bachelor's line; scholarship and hackathons to NYU.
  honors: (c) => {
    const t = mk(c);
    const p = c.mobile ? [8, 15] : [24, 30];
    return {
      ipu: t.wrap(c.rect('h-research'), p[0], -1),
      tcs: t.side('L', 0),
      nyu: t.wrap(union(c.rect('h-scholar'), c.rect('h-hack')), p[1], 1),
    };
  },

  // Three thick bands carry the carousel.
  gallery: (c) => {
    const r = c.rect('carousel');
    const band = r.h / 3;
    const order = ['nyu', 'tcs', 'ipu'];
    const out = {};
    order.forEach((k, i) => {
      const y = r.top + band * (i + 0.5);
      out[k] = { pts: [[-c.w, y], [2 * c.w, y]], w: band, o: 0.82 };
    });
    return out;
  },

  // Back where it started: the lines pass through the mark again and leave.
  contact: (c) => mk(c).mark(c.rect('mark-end'), false),
};

/* ------------------------------ component ------------------------------ */

const ease = (x) => x * x * (3 - 2 * x);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

// Feel of the strings.
const STIFFNESS = 0.13; // pull toward where the line should be
const DAMPING = 0.74; // how quickly the wobble dies
const DRIFT = 2.4; // px of slow idle movement
const REACH = 150; // px around the cursor that pushes the lines
const PUSH = 46; // px a line moves when the cursor is right on it

/**
 * rootRef   — element containing the <section data-pose> blocks
 * onSection — called with the index of the section currently in view
 * focusRef  — ref whose .current is 'ipu' | 'tcs' | 'nyu' | null; the named
 *             line is emphasised and the others fade back
 * intro     — play the draw-in animation
 */
export default function Lines({ rootRef, onSection, focusRef, intro }) {
  const pathRefs = useRef([]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return undefined;

    let poses = [];
    let tops = [];
    let state = null; // per line: position, velocity, width, opacity
    let raf = 0;
    let lastIdx = -1;
    let mx = -1e5;
    let my = -1e5;
    const t0 = performance.now();
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    const measure = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const secs = [...root.querySelectorAll('[data-pose]')];
      tops = secs.map((s) => s.getBoundingClientRect().top + window.scrollY);
      poses = secs.map((s) => {
        const sr = s.getBoundingClientRect();
        const rect = (name) => {
          const el = s.querySelector(`[data-a="${name}"]`);
          const r = el.getBoundingClientRect();
          const left = r.left - sr.left;
          const top = r.top - sr.top;
          return { left, top, right: left + r.width, bottom: top + r.height, w: r.width, h: r.height, cx: left + r.width / 2, cy: top + r.height / 2 };
        };
        const spec = POSES[s.dataset.pose]({ w, h, rect, mobile: w < 768 });
        return KEYS.map((k) => ({ pts: resample(spec[k].pts, N), w: spec[k].w, o: spec[k].o, soft: spec[k].soft ?? 0 }));
      });
      schedule();
    };

    const frame = (now) => {
      raf = 0;
      if (!poses.length) return;
      const y = window.scrollY;
      let i = 0;
      while (i < tops.length - 1 && y >= tops[i + 1]) i++;
      const last = i >= tops.length - 1;
      const f = last ? 0 : clamp01((y - tops[i]) / (tops[i + 1] - tops[i]));
      const A = poses[i];
      const B = poses[last ? i : i + 1];
      const offA = tops[i] - y;
      const offB = last ? offA : tops[i + 1] - y;

      const idx = f > 0.5 ? i + 1 : i;
      if (idx !== lastIdx) { lastIdx = idx; onSection?.(idx); }

      const first = !state;
      if (first) state = KEYS.map(() => ({ pos: new Float32Array(N * 2), vel: new Float32Array(N * 2), w: 0, o: 0 }));
      const time = (now - t0) / 1000;
      const focus = focusRef?.current ? KEYS.indexOf(focusRef.current) : -1;
      const reach2 = REACH * REACH;
      let moving = 0;

      for (let l = 0; l < KEYS.length; l++) {
        const a = A[l];
        const b = B[l];
        const c = state[l];
        // The three lines leave one after another, so a transition reads as a
        // ribbon unspooling rather than one block moving.
        const t = ease(clamp01((f - 0.08 - l * 0.05) / 0.72));
        const soft = reduced ? 0 : a.soft + (b.soft - a.soft) * t;
        let d = '';
        for (let p = 0; p < N; p++) {
          const ix = 2 * p;
          const iy = ix + 1;
          let tx = a.pts[ix] + (b.pts[ix] - a.pts[ix]) * t;
          let ty = a.pts[iy] + offA + (b.pts[iy] + offB - a.pts[iy] - offA) * t;
          if (soft > 0) {
            tx += Math.sin(p * 0.028 + time * 0.9 + l * 2.1) * DRIFT * soft;
            ty += Math.cos(p * 0.024 + time * 0.7 + l * 1.3) * DRIFT * soft;
            const dx = tx - mx;
            const dy = ty - my;
            const d2 = dx * dx + dy * dy;
            if (d2 < reach2) {
              const dist = Math.sqrt(d2) || 1;
              const fall = 1 - dist / REACH;
              const push = fall * fall * PUSH * soft;
              tx += (dx / dist) * push;
              ty += (dy / dist) * push;
            }
          }
          if (first || reduced) {
            c.pos[ix] = tx;
            c.pos[iy] = ty;
          } else {
            c.vel[ix] = (c.vel[ix] + (tx - c.pos[ix]) * STIFFNESS) * DAMPING;
            c.vel[iy] = (c.vel[iy] + (ty - c.pos[iy]) * STIFFNESS) * DAMPING;
            c.pos[ix] += c.vel[ix];
            c.pos[iy] += c.vel[iy];
            const m = Math.abs(c.vel[ix]) + Math.abs(c.vel[iy]);
            if (m > moving) moving = m;
          }
          // Draw through the midpoints with each point as a control point:
          // curves come out smooth instead of faceted, corners stay tight.
          const px = c.pos[ix].toFixed(1);
          const py = c.pos[iy].toFixed(1);
          if (p === 0) d = `M${px} ${py}`;
          else if (p === N - 1) d += `L${px} ${py}`;
          else d += `Q${px} ${py} ${((c.pos[ix] + c.pos[ix + 2]) / 2).toFixed(1)} ${((c.pos[iy] + c.pos[iy + 2]) / 2).toFixed(1)}`;
        }
        // Width blends geometrically so a line swelling into a full-page wash
        // stays a line for most of the transition.
        let tw = a.w ** (1 - t) * b.w ** t;
        let to = a.o + (b.o - a.o) * t;
        if (focus >= 0) {
          if (focus === l) { if (tw < 14) tw *= 1.7; to = Math.min(1, to * 1.6); } else to *= 0.22;
        }
        const k = first || reduced ? 1 : 0.2;
        c.w += (tw - c.w) * k;
        c.o += (to - c.o) * k;
        const el = pathRefs.current[l];
        if (el) {
          el.setAttribute('d', d);
          el.setAttribute('stroke-width', c.w.toFixed(1));
          el.setAttribute('stroke-opacity', c.o.toFixed(3));
        }
      }
      // With motion allowed the lines never fully rest (idle drift), so keep
      // the loop running while the tab is visible.
      if (!reduced || moving > 0.05) schedule();
    };

    function schedule() {
      if (!raf && !document.hidden) raf = requestAnimationFrame(frame);
    }

    const onMove = (e) => {
      if (e.pointerType !== 'mouse') return;
      mx = e.clientX;
      my = e.clientY;
    };
    const onLeave = () => { mx = -1e5; my = -1e5; };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(root);
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('pointermove', onMove, { passive: true });
    document.documentElement.addEventListener('pointerleave', onLeave);
    document.addEventListener('visibilitychange', schedule);
    document.fonts?.ready.then(measure);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('pointermove', onMove);
      document.documentElement.removeEventListener('pointerleave', onLeave);
      document.removeEventListener('visibilitychange', schedule);
    };
  }, [rootRef, onSection, focusRef]);

  return (
    <svg className={`st-lines${intro ? ' is-intro' : ''}`} aria-hidden="true">
      {KEYS.map((k, i) => (
        <path
          key={k}
          ref={(el) => { pathRefs.current[i] = el; }}
          pathLength="1"
          fill="none"
          stroke={LINE_COLORS[k]}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}
    </svg>
  );
}

// Static version of the mark for the nav and favicon-sized uses.
export const Mark = ({ size = 28, color = 'currentColor', strokeWidth = 9 }) => (
  <svg width={size} height={size} viewBox="-110 -110 220 220" aria-hidden="true">
    <path
      d={MARK_POINTS.map(([x, y], i) => `${i ? 'L' : 'M'}${x * 100} ${y * 100}`).join(' ')}
      fill="none"
      stroke={color}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
