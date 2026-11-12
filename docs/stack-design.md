# Wallet v2 — Drawer + Metallic Blueprint

This is the source of truth for the `/v2` portfolio. When new cards or
sub-decks are added later, every addition must obey the rules here so the
visual language stays coherent.

## Metaphor

A **drawer of cards**. Cards are stacked at the bottom of the screen with
no 3D rotation. Each upcoming card behind the active one shows only its
**label strip** (~32px tall) above the card in front. The active card sits
at the bottom with its full closed face visible (label + title + subtitle).
Click → the active card grows in place to full detail height with a
scrollable body. Cards already passed sink slightly under the active card
and fade.

Above the stack: a **chat bar** with the panda avatar and one of three
modes (collapsed / expanded / reply).

## Aesthetic

- Pure background `#0a0807` (warm almost-black, not pure black).
- Two soft radial vignettes (warm gold top, deep umber bottom) — candlelit feel.
- All ten card faces are **metallic gradients** (graphite, emerald, bronze,
  cobalt, copper, amethyst, titanium, brass, ruby, gold). Each has its own
  ink (text), mute (secondary), and accent.
- Surface treatments: top-left specular highlight, top edge vignette, fine
  noise grain, brushed-metal vertical streaks. All baked into `Card.jsx`.
- Type:
  - **Inter Tight** for prose, titles, body
  - **JetBrains Mono** for labels, dates, status microcopy
- No gradients on top of card faces. No glassmorphism. No paper / glass
  surface variants in v1 — metal only.

## Layout

```
┌──────────────────────────────────────────┐
│  CHAT BAR  (10/50/14% by mode)            │
│  panda · input · transcript / starters     │
├──────────────────────────────────────────┤
│                                            │
│   peek tab: PROJECTS                       │
│   peek tab: EXPERIENCE                     │
│   peek tab: JOURNEY                        │
│  ┌──────────────────────────┐              │
│  │ NOW                      │              │
│  │ This week                │  ← active    │
│  │ What I'm building        │              │
│  └──────────────────────────┘              │
│                                            │
│  [Bhavesh]            [Open to work]       │
└──────────────────────────────────────────┘
```

Right-edge rail: vertical progress dots, one per card, the active one is a
22px-tall pill.

When a card is **focused** (clicked open):
- A dim overlay covers the rest of the stack (`rgba(0,0,0,0.55)` + `blur(3px)`).
- Active card grows in place to ~`min(640, viewportH - 110)` tall.
- Header sticks at top showing label + title + subtitle in larger type.
- Body scrolls internally.
- Chrome (pills, dots, hint) hides.
- Click outside / Esc / scroll past content top closes (returns to stack).

## Cards

Order in the drawer top → bottom (the order the user sees scrolling forward):

| # | id | Material | Label |
|---|---|---|---|
| 0 | about | graphite | ABOUT |
| 1 | now | emerald | NOW |
| 2 | journey | bronze | JOURNEY |
| 3 | experience | cobalt | EXPERIENCE |
| 4 | skills | titanium | SKILLS |
| 5 | projects | copper | PROJECTS |
| 6 | design | amethyst | DESIGN |
| 7 | publications | brass | PUBLICATIONS |
| 8 | leadership | ruby | LEADERSHIP |
| 9 | contact | gold | CONTACT |

When the Projects sub-deck lands later, its child cards should use a
**tonal copper family** (dark bronze, copper, rust, oxidised orange) so the
visual shift between parent deck and sub-deck is felt — home is rainbow,
sub-deck is monochromatic.

## Card geometry

| Prop | Value |
|---|---|
| width (desktop) | 320 → 560px (responsive based on viewport) |
| height (closed/active) | 220px |
| height (focused) | `min(640, stackZoneH - 110)` |
| border radius | 18px |
| transform-origin | n/a — no 3D |
| peek tab height | 32px |
| `bottom` (active) | 0 |
| `bottom` (upcoming card #d) | `220 + (d - 1) * 32` |

Animations are pure CSS transitions on `bottom`, `height`, `opacity` using
`cubic-bezier(0.32, 0.72, 0, 1)` (Apple's cubic-out) at 700ms.

## Scroll behaviour

**Stack mode** (no card focused):
- Wheel / touch / arrow / page-up-down: one input → one card advance, with
  a 600ms cooldown to defeat trackpad inertia.
- Click a peek tab: it becomes active.
- Click the active card: it focuses (opens).

**Focused mode**:
- Internal body scroll first.
- At content bottom, continued downward wheel accumulates "peek-down"
  overflow. Past `PEEK_UNLOCK + COMMIT_PX` (=140px) the stack commits to
  the next card and that card focuses in the same slot.
- Symmetric upward at content top → previous card.
- Click overlay / Esc: closes the focused card.

## Chrome (visible only in stack mode)

| Element | Position | Purpose |
|---|---|---|
| Identity pill | bottom-left | panda + name + "AI · ENG" |
| Availability pill | bottom-right | green dot + "Open to Summer 2026" |
| Progress dots | right rail | jumpable per-card indicators |
| Scroll hint | top-left | one-time "↓ scroll to browse · click to open" |

Hidden when a card is focused — the detail view is uncluttered.

## Chat bar — three modes

| Mode | Height | Trigger | Contents |
|---|---|---|---|
| **collapsed** | 10% (12% mobile) | default | panda + small input + "hover to expand" hint |
| **expanded** | 50% (65% mobile) | mouse-enter / focus / `isStreaming` | header, scrollable transcript, tool-call chips, contextual starter chips, large input + send button |
| **reply** | 14% (18% mobile) | `lastAgentMsg && openMode` | one-line strip of the last assistant message; click to dismiss |

Streaming choreography (option H-1):
1. User sends a question → bar enters **expanded**.
2. Tool-call chips appear as `tool_call` events arrive.
3. `ui_command` arrives → card opens to focused state under the dim overlay,
   target subsection scrolls to + pulses, highlight phrases pulse-mark in
   place.
4. `text_delta` events stream into the latest bot bubble (live caret).
5. `final` event fires → message text finalises, `lastAgentMsg` is set,
   bar collapses to **reply** mode showing the last sentence as a strip.
6. User clicks the strip → reply dismissed, bar back to **collapsed**.

## Avatar

`/panda.png` rendered as a circular image, **no ring around it**. Always
moves subtly:

- **idle** — gentle breathing scale (1 → 1.04 → 1) at 3.6s, no glow
- **thinking** — faster breathe (1.4s) + tiny wobble + amber accent dot pulse
- **searching_profile** — slight tilt cycle + blue dot pulse
- **searching_web** — opposite tilt + emerald dot pulse
- **speaking** — small head bob + rose dot pulse
- **done** — settles, emerald dot, no pulse
- **error** — still, red dot pulse

A small status dot sits at the bottom-right corner (8-12px) coloured by
state. Replaces the gold "B" badge from the mockup with the actual panda
asset.

## Streaming + agent contract (preserved from prior v2)

No backend changes from the prior phase. The agent still emits SSE frames:

```
{type: "thread", thread_id: "..."}
{type: "status", stage: "thinking"|"searching_profile"|"searching_web"|"speaking"|"done"}
{type: "tool_call", name: "search_profile"|"search_web", query: "...", role?: "..."}
{type: "ui_command", action: "SET_ACTIVE_CARD",
  payload: {cardId, sectionId?, highlight?: [string, ...]}}
{type: "text_delta", content: "..."}
{type: "component", data: {...}}
{type: "final", answer, source, scroll_to, focus}
```

Frontend interprets `ui_command` by calling `applyFocusCommand()` on the
context, which navigates `/v2/:cardId/open` (deep link sync), scrolls to
`sectionId` inside the card, and pulses any `highlight` phrases.

## Now card

The only card whose content is fetched live. `GET /now` reads from
`backend/now.json` so updating "what I'm building" is just editing JSON
and the next request picks it up — no redeploy. Falls back to a static
default if the request fails (offline backend).

## Deep links

| URL | Meaning |
|---|---|
| `/v2` | home deck, no card focused |
| `/v2/:cardId` | home deck, that card focused in the drawer |
| `/v2/:cardId/open` | home deck, that card opened in detail |

When a portal deck (Projects sub-deck) is added: `/v2/projects` opens that
sub-deck; `/v2/projects/:projectId/open` opens its detail.

## Performance

- `will-change: transform, top, height` on every card (only ~10 of them, fine).
- No layout-id morphing, no perspective, no expensive 3D math.
- `prefers-reduced-motion`: disables phrase pulse + section ring + panda
  micro-motion, keeps positional CSS transitions.

## Mobile (deferred polish)

Desktop-first per spec. When mobile pass lands:
- Card width drops to ~92vw with a 380px max.
- ChatBar collapsed height bumps to 12% (already done).
- Tap-to-expand replaces hover-to-expand on the chat bar (already done).
- Progress-dot rail hides on small viewports (already done).
- Touch swipe drives the stack already.

## What's deferred

- Projects portal → per-project sub-deck (`kind: 'leaf' → 'portal'`)
- Sub-deck breadcrumb + back transition
- Mobile recalibration polish pass
- Sound effects
