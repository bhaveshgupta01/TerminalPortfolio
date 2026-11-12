# Portfolio setup

## Run it

```bash
npm install
cp .env.local.example .env.local   # then paste your ANTHROPIC_API_KEY
npm run dev                        # http://localhost:5173
```

The site works without a key; the chat then replies with a short "not switched on" message.

## Routes

| URL | What |
|---|---|
| `/` | Current site: the three-line scroll page (`src/pages/Story.jsx`) |
| `/v1` | Original scrolling portfolio (`src/pages/Home.jsx`) |
| `/v2` | Metallic card stack |
| `/wallet`, `/lab` | Earlier card-stack drafts |
| `/design` | Full design gallery |

## Editing content

Everything on the home page comes from `src/data/story.js`. The chat assistant builds
its knowledge from the same file (`api/_profile.js`), so one edit updates both.

## Chat

`api/chat.js` is a single serverless function. Vercel deploys it with the site; in
development Vite serves it through a small middleware in `vite.config.js`. There is
no database and no second server.

Deploy checklist:

1. Vercel > Project > Settings > Environment Variables: add `ANTHROPIC_API_KEY`.
2. Optional: add `CHAT_MODEL` to pick a different Claude model.
3. Anthropic Console > Limits: set a monthly spend limit. This is the hard ceiling on cost.
4. Redeploy, then ask the chat a question on the live site.

Built-in protection: the key never reaches the browser, questions are capped at 600
characters, replies are capped in length, and each visitor is limited to 12 questions
per five minutes (per server instance, so treat it as a speed bump, not a wall).

## Seeing what visitors ask

Every question is written to the function logs as a `[chat-log]` line (Vercel > Project >
Logs). Vercel only keeps those for a short time, so for a permanent record connect a
database:

1. Vercel > Project > Storage > Create > **Upstash for Redis** (free tier), connect it to
   this project. Vercel adds `KV_REST_API_URL` and `KV_REST_API_TOKEN` for you.
2. Add an `ADMIN_KEY` environment variable: any long random string.
3. Redeploy. Your private log is at `/api/questions?key=<your ADMIN_KEY>`.

With the database connected, the rate limit is also shared across all server instances.
Visitors' IP addresses are not stored. The chat tells visitors that questions are saved.

## Archived agent (`backend/`)

The FastAPI + LangGraph + ChromaDB agent that powered the chat in `/v1` and `/v2` is
kept in `backend/` but no longer deployed, so the chat in those versions shows its
offline message. See `backend/README.md` if you want to run it locally.
