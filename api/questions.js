// GET /api/questions?key=<ADMIN_KEY>
// A private page listing what visitors have asked the chat, newest first.
// Needs ADMIN_KEY set, and a Redis database connected (see SETUP.md).

import { hasStore, recent } from './_store.js';

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const text = (status, body) => new Response(body, { status, headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' } });

export async function GET(request) {
  const key = new URL(request.url).searchParams.get('key') || '';
  const admin = process.env.ADMIN_KEY || '';
  if (!admin || key.length !== admin.length || key !== admin) return text(404, 'Not found');
  if (!hasStore) return text(501, 'No database connected yet, so questions are only in the Vercel function logs (search for "[chat-log]"). See SETUP.md.');

  let rows;
  try {
    rows = await recent(500);
  } catch (err) {
    return text(502, `Could not read the log: ${err.message}`);
  }

  const body = rows.map((r) => `<tr><td>${esc((r.at || '').replace('T', ' ').slice(0, 16))}</td><td>${esc(r.question)}</td><td>${esc(r.answer)}</td><td>${esc(r.status)}</td></tr>`).join('');
  const html = `<!doctype html><meta charset="utf-8"><meta name="robots" content="noindex"><title>Chat questions</title>
<style>body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#17150f;background:#f5f2ea}table{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid #d9d4c7}td:first-child{white-space:nowrap;color:#6a6558}td:nth-child(3){color:#6a6558}</style>
<h1>Chat questions (${rows.length})</h1>
<table><tr><th>When (UTC)</th><th>Question</th><th>Answer</th><th>Status</th></tr>${body}</table>`;
  return new Response(html, { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' } });
}
