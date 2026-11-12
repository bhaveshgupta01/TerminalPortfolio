/**
 * Dark-metallic surfaces for the v2 wallet drawer.
 *
 * Mental model: each card is a slab of polished dark metal lit from above.
 * The gradient stops are weighted toward the deep core (38% mark) and the
 * deep base (72% mark) — those are the bulk of the card's visible mass and
 * they need to be DARK. The bright highlight stop (0%) and the specular
 * glaze are what give the metal its shine.
 *
 *   bg    — 4-stop linear gradient, 155° (highlight → midtone → deep → midtone)
 *   sheen — radial pass at top giving the "rolled metal" feel
 *   specular — sharp top-left glaze (the directional shine point)
 *   bevel — bright edge band at very top (the machined lip)
 *   base  — bottom shadow seal (anchors the card to the surface)
 *   ink   — primary text colour (very bright on dark metal)
 *   tag   — uppercase label colour, brighter than mute so labels stay legible
 *   mute  — secondary text (subtitles, dates)
 *   accent — call-out / link / pulse colour, tuned per metal
 *   tint  — single midtone hex used by the ambient background
 */

const M = (cfg) => ({
  bg: cfg.bg,
  sheen: cfg.sheen,
  specular: cfg.specular,
  tint: cfg.tint,
  ink: cfg.ink,
  tag: cfg.tag,
  mute: cfg.mute,
  accent: cfg.accent,
  bevel: cfg.bevel || 'linear-gradient(180deg, rgba(255,255,255,0.36) 0%, rgba(255,255,255,0) 8%, transparent 100%)',
  base:  cfg.base  || 'linear-gradient(0deg, rgba(0,0,0,0.50) 0%, transparent 28%)',
});

export const MATERIALS = {
  // gunmetal — anthracite blacks with a brushed steel highlight
  graphite: M({
    bg: 'linear-gradient(155deg, #5a5e6a 0%, #1c1e23 38%, #06080b 72%, #1a1c20 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(255,255,255,0.32) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(255,255,255,0.40) 0%, rgba(255,255,255,0) 26%)',
    tint: '#3c3f47', ink: '#f6f7f9', tag: '#cfd3da', mute: '#8a8e98', accent: '#d8dce4',
  }),
  // forest steel — deep evergreen
  emerald: M({
    bg: 'linear-gradient(155deg, #4e8e74 0%, #173c2c 38%, #06180e 72%, #163f2c 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(180,255,210,0.34) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(190,255,220,0.42) 0%, rgba(190,255,220,0) 26%)',
    tint: '#1f4634', ink: '#effff5', tag: '#a8e0c0', mute: '#6ea088', accent: '#9be8c0',
  }),
  // oxidized bronze — warm dark earth
  bronze: M({
    bg: 'linear-gradient(155deg, #b48050 0%, #5e3c1c 38%, #20120a 72%, #4e3018 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(255,225,180,0.36) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(255,225,180,0.42) 0%, rgba(255,225,180,0) 26%)',
    tint: '#603e1c', ink: '#fff0d8', tag: '#e0bb86', mute: '#9c7a52', accent: '#f0c590',
  }),
  // deep navy steel — dark cobalt
  cobalt: M({
    bg: 'linear-gradient(155deg, #5573c8 0%, #1a2a5e 38%, #060c1f 72%, #1a2c5a 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(195,225,255,0.32) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(195,225,255,0.42) 0%, rgba(195,225,255,0) 26%)',
    tint: '#1f316a', ink: '#f0f5ff', tag: '#b8c8ec', mute: '#8090b8', accent: '#c4d6f4',
  }),
  // dark copper — fired metal
  copper: M({
    bg: 'linear-gradient(155deg, #d68658 0%, #6e3a1c 38%, #260e06 72%, #5c331a 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(255,210,170,0.38) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(255,210,170,0.46) 0%, rgba(255,210,170,0) 26%)',
    tint: '#6e3a1c', ink: '#fff0dc', tag: '#f0b58c', mute: '#aa7758', accent: '#ffba90',
  }),
  // deep amethyst — dark plum
  amethyst: M({
    bg: 'linear-gradient(155deg, #8a62c0 0%, #2e1858 38%, #15082c 72%, #2c1652 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(235,205,255,0.34) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(235,205,255,0.44) 0%, rgba(235,205,255,0) 26%)',
    tint: '#321a5c', ink: '#f8edff', tag: '#d2b0ec', mute: '#9778b8', accent: '#e0c4ff',
  }),
  // gunmetal silver — darker titanium
  titanium: M({
    bg: 'linear-gradient(155deg, #98a0b0 0%, #3a3d44 38%, #181a1f 72%, #383b42 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(225,230,240,0.32) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(225,230,240,0.40) 0%, rgba(225,230,240,0) 26%)',
    tint: '#3d4048', ink: '#f5f6fa', tag: '#cdd2dc', mute: '#9a9fa9', accent: '#dadde4',
  }),
  // deep brass
  brass: M({
    bg: 'linear-gradient(155deg, #c8a050 0%, #5e4814 38%, #20180a 72%, #58420f 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(255,238,180,0.36) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(255,238,180,0.42) 0%, rgba(255,238,180,0) 26%)',
    tint: '#62491a', ink: '#fff5d8', tag: '#e8c878', mute: '#a08856', accent: '#f0d68a',
  }),
  // deep oxblood ruby
  ruby: M({
    bg: 'linear-gradient(155deg, #c0586c 0%, #5a1c30 38%, #1e0810 72%, #561a2c 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(255,210,225,0.36) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(255,210,225,0.44) 0%, rgba(255,210,225,0) 26%)',
    tint: '#5e1e34', ink: '#fce8ee', tag: '#f0a8c0', mute: '#a86c84', accent: '#ffb8d0',
  }),
  // rich dark gold
  gold: M({
    bg: 'linear-gradient(155deg, #e0a838 0%, #6e4e10 38%, #261c08 72%, #6c4c10 100%)',
    sheen: 'radial-gradient(ellipse 130% 70% at 50% -8%, rgba(255,232,160,0.42) 0%, transparent 56%)',
    specular: 'linear-gradient(135deg, rgba(255,232,160,0.48) 0%, rgba(255,232,160,0) 26%)',
    tint: '#6e4e10', ink: '#fff5cf', tag: '#f0cc6c', mute: '#a88848', accent: '#ffd970',
  }),
};

export const getMaterial = (key) => MATERIALS[key] || MATERIALS.graphite;
