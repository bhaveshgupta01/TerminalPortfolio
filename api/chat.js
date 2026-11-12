// POST /api/chat  { question: string, history?: [{ role: 'user'|'assistant', text: string }] }
//   -> 200 { answer, section, project, component }
//   -> 4xx/5xx { error: 'code', answer: 'friendly fallback text' }
// Every question is recorded through api/_store.js.
//
// Runs as a Vercel Function in production and through the Vite dev middleware
// locally (see vite.config.js). The only setup it needs is ANTHROPIC_API_KEY.

import Anthropic from '@anthropic-ai/sdk';
import { SYSTEM_PROMPT, SECTION_IDS, PROJECT_IDS } from './_profile.js';
import { record, overLimit } from './_store.js';

const MODEL = process.env.CHAT_MODEL || 'claude-opus-5-5';

// Hard limits that keep one visitor (or one script) from running up the bill.
const MAX_QUESTION_CHARS = 600;
const MAX_HISTORY_TURNS = 8;
const MAX_HISTORY_CHARS = 1500;
const MAX_OUTPUT_TOKENS = 4000; // includes the model's thinking; answers are short
const RATE_WINDOW_SECONDS = 5 * 60;
const RATE_MAX = 12; // requests per visitor per window

const EMAIL = 'bg2896@nyu.edu';
const FALLBACK = {
  not_configured: `The chat isn't switched on yet. Everything is on the page, or email me at ${EMAIL}.`,
  rate_limited: `That's a lot of questions in a short time. Give it a few minutes, or email me at ${EMAIL}.`,
  bad_request: 'I could not read that question. Try a shorter one?',
  refused: `That's not something I can answer here. Ask me about my work, or email me at ${EMAIL}.`,
  unavailable: `I can't reach my assistant right now. Everything is on the page, or email me at ${EMAIL}.`,
};

const OUTPUT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['answer', 'section', 'project', 'component'],
  properties: {
    answer: { type: 'string' },
    section: { type: 'string', enum: [...SECTION_IDS, 'none'] },
    project: { type: 'string', enum: [...PROJECT_IDS, ''] },
    component: {
      anyOf: [
        { type: 'null' },
        {
          type: 'object',
          additionalProperties: false,
          required: ['type', 'title', 'items'],
          properties: {
            type: { type: 'string', enum: ['stats', 'timeline', 'cards', 'list'] },
            title: { type: 'string' },
            items: {
              type: 'array',
              items: {
                type: 'object',
                additionalProperties: false,
                required: ['title', 'detail', 'meta'],
                properties: {
                  title: { type: 'string' },
                  detail: { type: 'string' },
                  meta: { type: 'string' },
                },
              },
            },
          },
        },
      ],
    },
  },
};

const reply = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
  });

const fail = (status, code) => reply(status, { error: code, answer: FALLBACK[code] });

export async function POST(request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return fail(400, 'bad_request');
  }
  const question = typeof body?.question === 'string' ? body.question.trim() : '';
  if (!question || question.length > MAX_QUESTION_CHARS) return fail(400, 'bad_request');

  // With a database connected this limit is shared across all instances;
  // without one it is per instance, and the spend limit set in the Anthropic
  // Console is the real ceiling.
  const ip = (request.headers.get('x-forwarded-for') || 'local').split(',')[0].trim();
  if (await overLimit(ip, RATE_MAX, RATE_WINDOW_SECONDS)) return fail(429, 'rate_limited');

  // Every question is recorded (without the visitor's IP), whatever happens next.
  const done = async (response, status, answer = '') => {
    await record({ question, answer: answer.slice(0, 600), status });
    return response;
  };

  if (!process.env.ANTHROPIC_API_KEY) return done(fail(503, 'not_configured'), 'not_configured');

  const history = (Array.isArray(body.history) ? body.history : [])
    .filter((m) => (m?.role === 'user' || m?.role === 'assistant') && typeof m.text === 'string' && m.text.trim())
    .slice(-MAX_HISTORY_TURNS)
    .map((m) => ({ role: m.role, content: m.text.slice(0, MAX_HISTORY_CHARS) }));
  while (history.length && history[0].role !== 'user') history.shift();

  const client = new Anthropic({ timeout: 25_000, maxRetries: 1 });

  try {
    const response = await client.beta.messages.create({
      model: MODEL,
      max_tokens: MAX_OUTPUT_TOKENS,
      // If the model declines a question on policy grounds, the API retries it
      // on a fallback model inside the same call.
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } }],
      output_config: { effort: 'low', format: { type: 'json_schema', schema: OUTPUT_SCHEMA } },
      messages: [...history, { role: 'user', content: question }],
    });

    if (response.stop_reason === 'refusal') return done(fail(200, 'refused'), 'refused');

    const text = response.content.find((b) => b.type === 'text')?.text ?? '';
    let out;
    try {
      out = JSON.parse(text);
    } catch {
      // Cut off at max_tokens or otherwise not valid JSON.
      return done(fail(502, 'unavailable'), 'bad_output');
    }

    const answer = String(out.answer || '').trim() || FALLBACK.unavailable;
    return done(reply(200, {
      answer,
      section: SECTION_IDS.includes(out.section) ? out.section : null,
      project: PROJECT_IDS.includes(out.project) ? out.project : null,
      component: out.component && Array.isArray(out.component.items) && out.component.items.length ? out.component : null,
    }), 'ok', answer);
  } catch (err) {
    if (err instanceof Anthropic.RateLimitError) return done(fail(429, 'rate_limited'), 'api_rate_limited');
    if (err instanceof Anthropic.AuthenticationError || err instanceof Anthropic.PermissionDeniedError) {
      console.error('[chat] API key rejected:', err.message);
      return done(fail(503, 'not_configured'), 'key_rejected');
    }
    if (err instanceof Anthropic.BadRequestError) {
      // Includes "credit balance too low" — log it so it shows in Vercel logs.
      console.error('[chat] request rejected:', err.message);
      return done(fail(502, 'unavailable'), 'api_rejected');
    }
    console.error('[chat] failed:', err?.message || err);
    return done(fail(502, 'unavailable'), 'error');
  }
}
