const API_BASE = import.meta.env.VITE_API_URL || '/api';

/**
 * Run hallucination detection on a query + response pair.
 */
export async function detectHallucination(query, response, contextDocs = null) {
  const res = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      response,
      context_docs: contextDocs,
    }),
  });
  if (!res.ok) throw new Error(`Detection failed: ${res.statusText}`);
  return res.json();
}

/**
 * Full chat flow: generate → detect → optionally auto-fix.
 */
export async function chatWithDetection(query, contextDocs = null, autoFix = false) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      context_docs: contextDocs,
      auto_fix: autoFix,
    }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.statusText}`);
  return res.json();
}

/**
 * Full chat flow using SSE for streaming generation, then detection.
 */
export async function* chatStreamWithDetection(query, contextDocs = null, autoFix = false, history = null) {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      context_docs: contextDocs,
      auto_fix: autoFix,
      history,
    }),
  });

  if (!res.ok) throw new Error(`Chat stream failed: ${res.statusText}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      
      if (chunk.startsWith("data: ")) {
        const dataStr = chunk.slice(6);
        try {
          const data = JSON.parse(dataStr);
          yield data;
        } catch (e) {
          console.error("Failed to parse chunk", chunk);
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

/**
 * Fix a hallucinated response.
 */
export async function correctResponse(query, response, unverifiedClaims = [], strategy = 'constrained', contextDocs = null) {
  const res = await fetch(`${API_BASE}/correct`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      response,
      unverified_claims: unverifiedClaims,
      context_docs: contextDocs,
      strategy,
    }),
  });
  if (!res.ok) throw new Error(`Correction failed: ${res.statusText}`);
  return res.json();
}

/**
 * Add documents to the knowledge base.
 */
export async function addToKnowledgeBase(documents, ids = null) {
  const res = await fetch(`${API_BASE}/kb/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents, ids }),
  });
  if (!res.ok) throw new Error(`KB add failed: ${res.statusText}`);
  return res.json();
}

/**
 * Health check.
 */
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}
