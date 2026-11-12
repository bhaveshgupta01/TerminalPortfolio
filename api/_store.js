// Optional durable storage for the chat: a question log and a shared rate
// limiter. Uses a Redis database over its REST API (Upstash, which is what
// Vercel's "Upstash for Redis" integration provisions). When no database is
// connected everything still works: questions go to the function logs only and
// the rate limiter falls back to per-instance memory.

const URL_ = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const TOKEN = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;

export const hasStore = Boolean(URL_ && TOKEN);

const LOG_KEY = 'chat:log';
const LOG_MAX = 5000;

async function redis(command) {
  const res = await fetch(URL_, {
    method: 'POST',
    headers: { authorization: `Bearer ${TOKEN}`, 'content-type': 'application/json' },
    body: JSON.stringify(command),
    signal: AbortSignal.timeout(2000),
  });
  if (!res.ok) throw new Error(`store ${res.status}`);
  return (await res.json()).result;
}

/** Save one exchange. Never throws: logging must not break the chat. */
export async function record(entry) {
  const row = { at: new Date().toISOString(), ...entry };
  console.log('[chat-log]', JSON.stringify(row));
  if (!hasStore) return;
  try {
    await redis(['LPUSH', LOG_KEY, JSON.stringify(row)]);
    await redis(['LTRIM', LOG_KEY, '0', String(LOG_MAX - 1)]);
  } catch (err) {
    console.error('[chat-log] store write failed:', err.message);
  }
}

/** Most recent exchanges, newest first. */
export async function recent(count = 200) {
  const rows = await redis(['LRANGE', LOG_KEY, '0', String(count - 1)]);
  return rows.map((r) => { try { return JSON.parse(r); } catch { return { question: String(r) }; } });
}

// In-memory fallback: each warm instance counts separately.
const hits = new Map();

/** True when this visitor has asked more than `max` questions in the window. */
export async function overLimit(ip, max, windowSeconds) {
  if (hasStore) {
    try {
      const key = `chat:rl:${ip}:${Math.floor(Date.now() / (windowSeconds * 1000))}`;
      const n = await redis(['INCR', key]);
      if (n === 1) await redis(['EXPIRE', key, String(windowSeconds)]);
      return n > max;
    } catch (err) {
      console.error('[chat] store rate limit failed, using memory:', err.message);
    }
  }
  const now = Date.now();
  if (hits.size > 5000) hits.clear();
  const recentHits = (hits.get(ip) || []).filter((t) => now - t < windowSeconds * 1000);
  recentHits.push(now);
  hits.set(ip, recentHits);
  return recentHits.length > max;
}
