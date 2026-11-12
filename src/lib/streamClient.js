/**
 * Thin SSE client for the lab agent endpoint.
 *
 * The backend emits `data: <json>\n\n` lines followed by a terminating
 * `data: [DONE]\n\n`. We use fetch + a ReadableStream reader so we can
 * preserve POST semantics (EventSource is GET-only) and keep the same
 * cookie/header story as the existing /chat call.
 *
 * Events fan out through `onEvent(evt)`. The caller decides how to route
 * each type (text_delta, ui_command, component, status, tool_call, final,
 * thread, error, done).
 */

export class StreamAbortError extends Error {
  constructor() {
    super('stream aborted');
    this.name = 'StreamAbortError';
  }
}

export async function streamChat({
  apiUrl,
  question,
  threadId,
  onEvent,
  signal,
}) {
  const res = await fetch(`${apiUrl}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ question, thread_id: threadId }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`stream ${res.status}: ${text || res.statusText}`);
  }
  if (!res.body) throw new Error('stream body missing');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = frame.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') {
          onEvent({ type: 'done' });
          return;
        }
        try {
          onEvent(JSON.parse(payload));
        } catch (err) {
          console.warn('[stream] bad JSON frame', payload, err);
        }
      }
    }
  } catch (err) {
    if (err?.name === 'AbortError') throw new StreamAbortError();
    throw err;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* reader already released */
    }
  }
}
