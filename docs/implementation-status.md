# Implementation Status

Snapshot date: `2026-04-21`

This document is the current technical handoff for the project. It records:

- what is already stable,
- what has been implemented after the stable baseline,
- how the current frontend and backend behave,
- what is intentionally deferred,
- and the recommended next implementation steps.

---

## 1. Repo and version status

### Stable baseline

- Stable release tag: `v0.1.0`
- Stable release commit: `4746e53`
- Stable backup branch on GitHub: `backup/v0.1.0`
- Stable branch reference locally: `main`

This baseline was created so the project can safely return to a known-good version if later experiments fail.

Related files:

- `VERSION`
- `CHANGELOG.md`
- `docs/versioning.md`

### Current active work branch

- Working branch: `feature/post-v0.1.0`

This branch contains the new Apple Wallet style frontend prototype work. At the time of this snapshot, that work exists locally on the feature branch and is not part of the stable `v0.1.0` release.

---

## 2. Stable system before the wallet refactor

Before the wallet UI work began, the project already had a working production-style portfolio system with the following shape.

### Frontend

- React 19
- Vite
- Tailwind CSS
- Framer Motion
- React Router

Routes before the wallet prototype:

- `/` → main scrolling portfolio
- `/design` → design gallery

The original homepage used:

- a full scrolling section-based portfolio,
- a floating Dynamic Island style chat interface,
- structured UI components rendered inside chat responses,
- a theme system with CSS variables,
- ambient animated background effects,
- a design gallery on a separate route.

### Backend

- FastAPI
- LangGraph
- ChromaDB
- Gemini-based agent

Current backend contract before streaming:

`POST /chat`

```json
{
  "answer": "string",
  "scroll_to": "about | experience | skills | projects | publications | leadership | contact | journey | null",
  "component": { "type": "...", "data": {} } | null,
  "source": "profile" | "web" | "both",
  "thread_id": "string"
}
```

Important current limitation:

- the backend returns one final JSON response,
- the frontend does not yet consume a streamed response,
- there is no UI command protocol yet,
- there is no partial text streaming yet.

### Dynamic chat component system

The frontend already supports structured chat responses through `DynamicComponent`, including:

- stats
- timeline
- cards
- comparison
- list
- code
- quote

This is important because the wallet refactor preserves this system rather than replacing it.

---

## 3. Wallet UI refactor goal

The goal of the wallet work is to convert the current scrolling portfolio into a fixed split-screen interface inspired by Apple Wallet:

- the top half is the AI conversation zone,
- the bottom half is a stacked card interface,
- the active card can be controlled by the user or by the AI,
- the AI and the visible UI should eventually feel synchronized.

The target interaction model is:

- user clicks a card → card opens,
- user scrolls inside the opened card → next card starts peeking,
- a larger continued scroll at the bottom moves to the next card,
- AI can focus a card when answering a question,
- later, the AI should also be able to stream partial text and UI commands in real time.

---

## 4. What has been implemented now

This section describes the current progress implemented after `v0.1.0`.

## 4.1 New wallet route

A separate route has been created so the wallet prototype can evolve without replacing the existing homepage yet.

New route:

- `/wallet`

Files:

- `src/App.jsx`
- `src/pages/WalletHome.jsx`

Why this was done:

- preserves the current stable homepage at `/`,
- allows side-by-side comparison of old vs new UX,
- reduces deployment and rollback risk during major UI experimentation.

---

## 4.2 Fixed split-screen layout

The wallet page now uses a fixed, non-scrolling viewport layout:

- top region: AI chat area
- bottom region: wallet stack area

Layout behavior:

- root container is full-height and non-scrolling,
- internal scrolling happens only inside the chat history and active wallet card,
- the page itself does not vertically scroll like the original homepage.

Implemented in:

- `src/pages/WalletHome.jsx`

Current layout split:

- top: approximately `40%`
- bottom: approximately `60%`

The implementation uses a flexible split based on container height rather than reusing the old section scroll model.

---

## 4.3 Wallet state layer

A dedicated wallet state layer has been added.

File:

- `src/context/WalletContext.jsx`

Current wallet state includes:

- `activeCard`
- `activeCardSource`
- `highlightedCard`
- `setActiveCard()`
- `clearActiveCard()`

### State meaning

`activeCard`

- currently opened wallet card id
- values include:
  - `about`
  - `journey`
  - `experience`
  - `skills`
  - `projects`
  - `leadership`
  - `publications`
  - `design`
  - `contact`

`activeCardSource`

- tracks who last opened the card:
  - `user`
  - `ai`
  - `route`

This is important because the current product rule is:

- if the user has explicitly selected a card, backend-driven focus should not immediately override that selection.

`highlightedCard`

- supports temporary visual emphasis when the AI focuses a card.

This creates the first bridge between agent output and UI response even before true stream commands exist.

---

## 4.4 Deep-linking support

The wallet page now supports URL-based card state using `?card=...`.

Examples:

- `/wallet?card=projects`
- `/wallet?card=experience`

Current behavior:

- opening the route with a valid `card` query opens that card,
- changing cards updates the query string,
- this does not materially impact response time because it is local client-side routing/state sync.

Implemented in:

- `src/pages/WalletHome.jsx`

This is useful for:

- sharing links,
- future AI-controlled navigation,
- future analytics,
- future “return user to exact card” behavior.

---

## 4.5 Wallet card content model

The wallet stack uses a card definition list instead of hard-coding each card directly inside the page.

File:

- `src/components/wallet/walletContent.jsx`

Each card definition currently contains:

- `id`
- `title`
- `eyebrow`
- `summary`
- `highlights`
- `gradient`
- `icon`
- `render`

This allows the stack component to stay generic while card content remains modular.

### Implemented cards

The wallet now includes:

- About
- Journey
- Experience
- Skills
- Projects
- Leadership
- Publications
- Design
- Contact

### Design card decision

The original `/design` route still exists, but the wallet now also includes a `Design` card that acts as a creative summary layer.

The `Design` card currently:

- previews selected design assets,
- shows category counts,
- exposes design tools used,
- links to the full `/design` route when deeper browsing is needed.

This satisfies the requirement to represent the design work inside the wallet without deleting the existing gallery route.

---

## 4.6 Wallet stack behavior

Core stack behavior is implemented in:

- `src/components/wallet/WalletStack.jsx`
- `src/components/wallet/WalletCard.jsx`

### Idle state

When `activeCard === null`:

- cards are stacked vertically,
- each card shares absolute positioning,
- each card offset is derived from its index,
- each deeper card is slightly smaller than the one above it.

Current motion logic:

- `y = index * 60`
- `scale = 1 - (index * 0.04)`
- descending `zIndex`

This reproduces the layered “wallet stack at rest” behavior.

### Active state

When a card is selected:

- it springs to the top,
- it becomes full scale,
- it becomes the front-most layer,
- its body becomes internally scrollable.

Current active behavior:

- `y = 0`
- `scale = 1`
- elevated `zIndex`
- inner content uses `overflow-y-auto`

### Hidden/background state

When a card is active:

- most non-selected cards move below the viewport,
- opacity fades out,
- the active card becomes the main readable surface.

This keeps the interaction focused rather than trying to show every card simultaneously.

### Peek behavior

A custom “next card peeking” behavior has been added to mimic the feel of Apple Wallet:

- while the user scrolls inside the active card,
- scroll progress is monitored,
- once the user gets near the bottom,
- the next card begins to rise and become visible.

Current implementation:

- peeking begins near the bottom of the current card,
- the next card transitions from hidden to partially visible,
- opacity and position interpolate based on inner scroll progress.

### Card-to-card scroll transitions

The stack supports:

- mouse wheel transitions,
- touch gesture transitions,
- internal card scrolling first,
- next-card jump only when the current card is at the boundary.

Current rule:

- at bottom + strong downward input → move to next card
- at top + strong upward input → move to previous card

This matches the requested “scroll through the wallet” behavior instead of forcing the user to close one card and open the next manually.

### Tap interactions

Current behavior:

- tap anywhere on a visible card to open it,
- tap outside the active card to close it.

This matches the product decisions made during clarification.

---

## 4.7 AI chat zone in the top half

The top half of the wallet layout is implemented in:

- `src/components/wallet/WalletChatPanel.jsx`
- `src/components/wallet/WalletAvatar.jsx`

### What was preserved

The new chat zone preserves:

- text conversation flow,
- suggestion chips,
- input box,
- typing indicator,
- structured `DynamicComponent` rendering,
- source badges for web/profile answers.

This was important because the chat system already works and should not be thrown away during the layout refactor.

### What changed

The old floating Dynamic Island behavior has been replaced by:

- a fixed top-half chat panel,
- a stateful avatar area,
- wallet-mode header treatment,
- chat history that scrolls only inside the top panel.

### Avatar states

The wallet avatar currently supports multiple UI states:

- idle
- thinking
- guiding
- replied

These states change glow and motion treatment so the UI can visually reflect what the agent is doing.

---

## 4.8 Current AI-to-UI bridge

This is the most important implementation detail of the first pass.

There is not yet a true stream parser, but the wallet already reacts to the current backend contract.

### Current bridge logic

In `src/pages/WalletHome.jsx`:

- the frontend still calls the existing `POST /chat`,
- waits for the final JSON response,
- reads `data.scroll_to`,
- maps that value into a wallet card focus event.

So today:

- backend returns `scroll_to: "projects"`
- frontend opens the `projects` wallet card

### Current safeguard

If the user has manually opened a card, the AI does not override it immediately.

Current rule:

- AI focus is allowed only when the user does not currently have an active user-selected card.

### Current highlight behavior

When AI focus occurs:

- the card can flash/highlight briefly,
- the user sees a small “AI focused” style signal on the card.

This is a temporary version of the future streamed UI synchronization model.

---

## 4.9 Preservation of existing backend behavior

No backend changes have been made yet for the wallet prototype.

That means:

- Railway deployment behavior is unchanged,
- FastAPI still serves the same `/chat` response shape,
- LangGraph output shape is unchanged,
- the frontend wallet version works with the backend exactly as it exists today.

This was intentional so the first pass could focus on UI architecture and interaction design before introducing protocol changes.

---

## 5. Files added or modified in this phase

### Added

- `src/context/WalletContext.jsx`
- `src/pages/WalletHome.jsx`
- `src/components/wallet/WalletAvatar.jsx`
- `src/components/wallet/WalletCard.jsx`
- `src/components/wallet/WalletStack.jsx`
- `src/components/wallet/WalletChatPanel.jsx`
- `src/components/wallet/walletContent.jsx`

### Modified

- `src/App.jsx`

---

## 6. What has been validated

Current validation completed:

- `npm run build`

Result:

- build passed successfully after the wallet route and components were added.

What has not yet been added:

- unit tests
- interaction tests
- end-to-end tests
- backend streaming tests
- mobile keyboard-specific QA

---

## 7. What is still intentionally not implemented

The wallet prototype is phase 1. The following features are still pending.

### 7.1 Partial text streaming

Not implemented yet.

Current state:

- frontend waits for one final JSON payload
- no token streaming or text delta handling exists

Future requirement:

- backend should stream partial text chunks,
- frontend should append text progressively,
- typing state should transition into actual streamed response state.

### 7.2 UI command protocol

Not implemented yet.

Current state:

- frontend only reacts to the old `scroll_to` field
- no dedicated command envelope exists

Future requirement:

- backend emits explicit UI commands, for example:

```json
{
  "type": "ui_command",
  "action": "SET_ACTIVE_CARD",
  "payload": {
    "cardId": "projects"
  }
}
```

This should become the long-term contract instead of relying only on `scroll_to`.

### 7.3 Stream parser on the frontend

Not implemented yet.

Future requirement:

- replace the final-only JSON handling with a stream reader,
- decode stream chunks,
- process text updates separately from UI commands and final component data.

Recommended future chunk types:

- `text_delta`
- `ui_command`
- `component`
- `final`
- `error`

### 7.4 Backend streaming endpoint

Not implemented yet.

Current backend:

- returns a normal `ChatResponse`

Future backend requirement:

- support SSE or NDJSON streaming from FastAPI,
- surface intermediate agent states,
- emit UI commands as part of the same response lifecycle.

### 7.5 Fine-grained highlighting inside a card

Not implemented yet.

Current behavior:

- AI can focus a card

Future behavior:

- AI should optionally highlight relevant subsections inside the active card,
- for example:
  - focus `projects`
  - highlight `VoiceGraph`

This likely requires:

- subsection ids,
- scroll-to-subsection support inside cards,
- richer command payloads.

### 7.6 Sub-card / subsection navigation

Not implemented yet.

Future idea:

- `SET_ACTIVE_CARD` can later expand to:

```json
{
  "type": "ui_command",
  "action": "SET_ACTIVE_CARD",
  "payload": {
    "cardId": "experience",
    "sectionId": "tcs",
    "highlight": ["2,600+ users", "99.2% uptime"]
  }
}
```

### 7.7 Mobile interaction polish

The current version is responsive and functional, but not fully polished to production depth.

Future refinement areas:

- keyboard-open behavior tuning,
- touch gesture tuning across devices,
- scroll threshold tuning,
- better small-screen card header compression,
- finer safe-area handling on iPhone.

### 7.8 Accessibility pass

Not implemented yet.

Future requirement:

- stronger keyboard navigation,
- focus trapping when needed,
- better ARIA states for active cards,
- reduced-motion fallbacks,
- screen-reader descriptions for AI focus events.

### 7.9 Analytics and behavior instrumentation

Not implemented yet.

Future requirement:

- track which cards are opened,
- track AI-driven focus events,
- compare classic homepage usage vs wallet usage,
- measure whether deep links improve navigation.

### 7.10 Deployment decision

The wallet UI exists as a separate route, but product rollout has not been finalized.

Future product decision:

- keep both `/` and `/wallet`,
- or promote `/wallet` to become the new homepage,
- or gate it behind an experiment flag.

---

## 8. Recommended next implementation roadmap

This is the recommended path after the current progress.

## Phase 2: streaming bridge

Goal:

- keep the new wallet UI,
- upgrade the AI experience from final-response-only to true synchronized streaming.

Recommended order:

1. Add a streaming response format on the backend.
2. Add a frontend stream parser.
3. Preserve compatibility with the old final JSON shape during migration.
4. Add `ui_command` events for card focus.
5. Add partial text streaming.
6. Add optional subsection highlight payloads.

### Recommended frontend parser shape

```js
if (chunk.type === "text_delta") {
  appendAssistantText(chunk.content);
}

if (chunk.type === "ui_command" && chunk.action === "SET_ACTIVE_CARD") {
  if (!userHasLockedCardSelection()) {
    setActiveCard(chunk.payload.cardId, { source: "ai", highlight: true });
  }
}

if (chunk.type === "component") {
  attachStructuredComponent(chunk.data);
}

if (chunk.type === "final") {
  finalizeMessage(chunk.data);
}
```

### Recommended backend event contract

Minimum recommended command structure:

```json
{
  "type": "ui_command",
  "action": "SET_ACTIVE_CARD",
  "payload": {
    "cardId": "projects"
  }
}
```

Future-compatible version:

```json
{
  "type": "ui_command",
  "action": "SET_ACTIVE_CARD",
  "payload": {
    "cardId": "projects",
    "sectionId": "voicegraph",
    "highlight": ["knowledge graph", "Gemini Live", "Neo4j"]
  }
}
```

## Phase 3: UI polish and product hardening

Recommended next improvements after streaming:

- tune Wallet motion thresholds to feel closer to Apple Wallet,
- add smoother card cover treatment while peeking,
- add subsection anchors inside large cards,
- add better transition between chat guidance and card reveal,
- add loading states tied to real stream stages,
- add tests for deep links, card selection, and AI focus rules,
- evaluate whether `/wallet` should become the default homepage.

---

## 9. Deployment impact summary

### Vercel

Once the current wallet frontend work is committed and pushed:

- Vercel will be able to serve `/wallet`
- the classic `/` route will still work
- no backend API contract changes are required for this first pass

### Railway

No new backend deployment is required for the current wallet prototype because:

- the API shape has not changed,
- the wallet version still consumes the existing `/chat` response.

Future streaming work will require backend updates and therefore a Railway redeploy.

---

## 10. Current product status in one sentence

The project now has a stable rollback release at `v0.1.0`, plus a separate Apple Wallet style frontend prototype on `feature/post-v0.1.0` that already supports user-driven card navigation and backend-driven card focus through the existing `scroll_to` contract, while true streaming and explicit UI command events remain the next major implementation phase.
