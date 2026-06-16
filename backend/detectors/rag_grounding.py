"""
RAG Grounding Detector — Web Search Edition

Grounds LLM response claims against live Google search results via Serper.dev,
instead of a static user-uploaded knowledge base.

Flow:
  1. Extract atomic factual claims from the LLM response (LLM call).
  2. For each claim, perform a real-time web search via Serper.dev.
  3. Aggregate top snippets as "context documents".
  4. Ask the LLM to classify each claim as supported / contradicted / not_found.

Graceful degradation:
  - If SERPER_API_KEY is missing → returns 0.5 (unverifiable penalty).
  - If a search request fails → that claim is treated as "not_found".
  - All HTTP calls are async (httpx.AsyncClient).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from llm_client import LLMClient
from detectors.base import BaseDetector, DetectorResult

# ── Prompts ──────────────────────────────────────────────────────────────────

_VERIFY_PROMPT = """\
You are a claim verification assistant. You will be given a list of claims \
and a set of web search result snippets.

For each claim, determine if it is:
- "supported"           — the snippets contain evidence that confirms the claim
- "partially_supported" — the snippets are related but don't fully confirm
- "contradicted"        — the snippets contain evidence that contradicts the claim
- "not_found"           — the snippets don't address this claim at all

Return ONLY a JSON object:
{
  "results": [
    {"claim": "...", "verdict": "supported|partially_supported|contradicted|not_found", "evidence_snippet": "..."},
    ...
  ]
}"""

_EXTRACT_CLAIMS_PROMPT = (
    "Extract all distinct, non-redundant factual claims from the text. "
    "Each claim must be unique — do not repeat the same fact in different words. "
    'Return ONLY a JSON object: {"claims": ["claim 1", "claim 2", ...]}'
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _dedup_claims(claims: list[str]) -> list[str]:
    """Remove duplicate claims by normalised (lowercased, stripped) text.

    Preserves the original casing of the first occurrence.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for claim in claims:
        key = claim.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(claim)
    return unique


# ── Web Search Retriever ──────────────────────────────────────────────────────

SERPER_URL = "https://google.serper.dev/search"


class WebSearchRetriever:
    """Fetches real-time Google results via the Serper.dev API."""

    def __init__(self, api_key: str, num_results: int = 5, http_client: httpx.AsyncClient | None = None):
        self._api_key = api_key
        self._num_results = num_results
        self._http_client = http_client

    async def search(self, query: str) -> list[dict[str, str]]:
        """Return a list of text snippets from the top organic results."""
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": self._num_results}

        try:
            if self._http_client:
                resp = await self._http_client.post(SERPER_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(SERPER_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    data: dict[str, Any] = resp.json()
        except Exception as exc:
            return [{
                "text": f"[Web search failed: {exc}]",
                "title": "Search Failed",
                "link": "https://google.com"
            }]

        results: list[dict[str, str]] = []

        # Answer box (direct answer) — highest signal
        if answer_box := data.get("answerBox"):
            for key in ("answer", "snippet", "snippetHighlighted"):
                if text := answer_box.get(key):
                    txt = text if isinstance(text, str) else " ".join(text)
                    results.append({
                        "text": txt,
                        "title": "Google Answer Box",
                        "link": answer_box.get("link", "https://google.com")
                    })

        # Knowledge graph snippet
        if kg := data.get("knowledgeGraph"):
            if desc := kg.get("description"):
                results.append({
                    "text": desc,
                    "title": kg.get("title", "Google Knowledge Graph"),
                    "link": kg.get("link", "https://google.com")
                })

        # Organic results
        for result in data.get("organic", [])[: self._num_results]:
            if snippet := result.get("snippet"):
                title = result.get("title", "Search Result")
                link = result.get("link", "https://google.com")
                results.append({
                    "text": f"{title}: {snippet}" if title else snippet,
                    "title": title,
                    "link": link
                })

        return results or [{
            "text": "[No web results returned]",
            "title": "No Results",
            "link": "https://google.com"
        }]


# ── Detector ─────────────────────────────────────────────────────────────────


class RAGGroundingDetector(BaseDetector):
    """Ground-truth verification against live web search results."""

    name = "rag_grounding"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        serper_api_key: str | None = None,
        top_k: int = 5,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.llm = llm_client or LLMClient()
        self.top_k = top_k

        key = serper_api_key or os.getenv("SERPER_API_KEY", "")
        self._retriever: WebSearchRetriever | None = (
            WebSearchRetriever(api_key=key, num_results=top_k, http_client=http_client) if key else None
        )

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def has_knowledge_base(self) -> bool:
        """Always True when a Serper key is configured (web is always available)."""
        return self._retriever is not None

    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] | None = None,
    ) -> DetectorResult:
        """Score groundedness against web search results or supplied context."""

        # If caller already supplies context docs, use those directly
        if context_docs:
            return await self._verify_against_docs(
                query, response, context_docs, source="context_docs"
            )

        if not self._retriever:
            return DetectorResult(
                name=self.name,
                score=0.5,
                evidence=[
                    "SERPER_API_KEY not configured — cannot perform web search.",
                    "Score defaulted to 0.5 (unverifiable penalty).",
                    "Set SERPER_API_KEY in backend/.env to enable web grounding.",
                ],
                metadata={"has_knowledge_base": False, "source": "none"},
            )

        # Step 1 — extract claims and deduplicate by normalised text
        claims = await self._extract_claims(response)
        claims = _dedup_claims(claims)
        if not claims:
            return DetectorResult(
                name=self.name,
                score=0.7,
                evidence=["No factual claims extracted from response."],
                metadata={"claims_extracted": 0, "source": "web_search"},
            )

        # Step 2 — search the web for each claim (concurrently)
        search_tasks = [self._retriever.search(claim) for claim in claims]
        per_claim_results: list[list[dict[str, str]]] = await asyncio.gather(*search_tasks)

        # Map claims to citation URLs
        claim_citations: dict[str, list[dict[str, str]]] = {}
        for claim, results in zip(claims, per_claim_results):
            claim_citations[claim] = [
                {"title": r["title"], "url": r["link"]}
                for r in results[:2] if r["link"]
            ]

        # Flatten into a deduplicated list (preserving order) of text snippets for context
        seen: set[str] = set()
        all_snippets: list[str] = []
        for results in per_claim_results:
            for r in results[:2]:  # take top 2 snippets per claim to prevent cutoff
                if r["text"] not in seen:
                    seen.add(r["text"])
                    all_snippets.append(r["text"])

        return await self._verify_against_docs(
            query,
            response,
            all_snippets,
            source="web_search",
            pre_extracted_claims=claims,
            claim_citations=claim_citations,
        )

    # ── internals ────────────────────────────────────────────────────────────

    async def _verify_against_docs(
        self,
        query: str,
        response: str,
        docs: list[str],
        source: str = "web_search",
        pre_extracted_claims: list[str] | None = None,
        claim_citations: dict[str, list[dict[str, str]]] | None = None,
    ) -> DetectorResult:
        """Ask the LLM to verify claims against reference documents/snippets."""

        claims = pre_extracted_claims or _dedup_claims(await self._extract_claims(response))
        if not claims:
            return DetectorResult(
                name=self.name,
                score=0.7,
                evidence=["No factual claims extracted from response."],
                metadata={"claims_extracted": 0, "source": source},
            )

        docs_text = "\n\n---\n\n".join(docs[: self.top_k * 2])
        claims_text = "\n".join(f"- {c}" for c in claims)

        verify_msg = (
            f"**Claims to verify:**\n{claims_text}\n\n"
            f"**Web search results:**\n{docs_text}"
        )

        result = await asyncio.to_thread(
            self.llm.chat_json,
            [
                {"role": "system", "content": _VERIFY_PROMPT},
                {"role": "user", "content": verify_msg},
            ],
            temperature=0.0,
        )

        if result.get("parse_error"):
            return DetectorResult(
                name=self.name,
                score=0.5,
                evidence=["Verification response could not be parsed."],
                metadata={"raw": result.get("raw", ""), "source": source},
            )

        verdicts = result.get("results", [])

        # Deduplicate verdicts by claim text — if the same claim appears
        # multiple times with conflicting verdicts, keep the most lenient one
        # (supported > partially_supported > not_found > contradicted).
        _VERDICT_RANK = {"supported": 0, "partially_supported": 1, "not_found": 2, "contradicted": 3}
        best_verdicts: dict[str, dict] = {}
        for v in verdicts:
            key = v.get("claim", "").strip().lower()
            if not key:
                continue
            existing = best_verdicts.get(key)
            if existing is None:
                best_verdicts[key] = v
            else:
                # Keep the verdict with the lower (more lenient) rank
                if _VERDICT_RANK.get(v.get("verdict", "not_found"), 2) < _VERDICT_RANK.get(existing.get("verdict", "not_found"), 2):
                    best_verdicts[key] = v

        verdicts = list(best_verdicts.values())

        supported = [v for v in verdicts if v.get("verdict") == "supported"]
        partial = [v for v in verdicts if v.get("verdict") == "partially_supported"]
        contradicted = [v for v in verdicts if v.get("verdict") == "contradicted"]
        not_found = [v for v in verdicts if v.get("verdict") == "not_found"]

        total = len(verdicts) or 1
        grounding_score = (
            len(supported) * 1.0
            + len(partial) * 0.5
            + len(contradicted) * 0.0
            + len(not_found) * 0.3
        ) / total

        grounding_score = max(0.0, min(1.0, grounding_score))

        verified = [v["claim"] for v in supported + partial]
        unverified = [v["claim"] for v in contradicted + not_found]

        evidence_snippets = [
            f'{v.get("claim", "?")} → {v.get("verdict", "?")} '
            f'(web evidence: {v.get("evidence_snippet", "N/A")})'
            for v in verdicts
        ]

        # Compile citations dictionary for verified claims
        citations = {}
        if claim_citations:
            for v in verdicts:
                verdict_str = v.get("verdict")
                if verdict_str in ("supported", "partially_supported"):
                    c_text = v.get("claim", "")
                    if c_text in claim_citations:
                        citations[c_text] = claim_citations[c_text]

        return DetectorResult(
            name=self.name,
            score=grounding_score,
            verified_claims=verified,
            unverified_claims=unverified,
            evidence=evidence_snippets,
            metadata={
                "source": source,
                "docs_retrieved": len(docs),
                "claims_extracted": len(claims),
                "supported": len(supported),
                "partially_supported": len(partial),
                "contradicted": len(contradicted),
                "not_found": len(not_found),
                "citations": citations,
            },
        )

    async def _extract_claims(self, text: str) -> list[str]:
        """Use the LLM to extract atomic factual claims from text."""
        result = await asyncio.to_thread(
            self.llm.chat_json,
            [
                {"role": "system", "content": _EXTRACT_CLAIMS_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        return result.get("claims", [])

    # ── kept for backward-compat (no-ops) ────────────────────────────────────

    def add_documents(self, documents: list[str], ids: list[str] | None = None):
        """Deprecated: Web search replaced static KB. This is a no-op."""
        pass

    def reset_knowledge_base(self):
        """Deprecated: No local KB to reset. This is a no-op."""
        pass