"""
NLI Entailment Detector

Uses the HuggingFace Inference API with facebook/bart-large-mnli to check
whether factual claims in the LLM response are logically entailed by the
query context — a fundamentally different signal from the LLM-based detectors.

Flow:
  1. Extract atomic claims from the response (via LLM, same prompt as RAG).
  2. For each claim, call HuggingFace zero-shot classification:
       premise   = query
       hypothesis = claim
  3. Model returns probabilities for: entailment / neutral / contradiction.
  4. entailment  → verified claim   (score contribution: 1.0)
     neutral     → uncertain claim  (score contribution: 0.4)
     contradiction → unverified claim (score contribution: 0.0)

Graceful degradation:
  - If HF_API_KEY is missing → returns 0.5 (unverifiable).
  - If the API call fails → logs and treats that claim as neutral.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from llm_client import LLMClient
from detectors.base import BaseDetector, DetectorResult

# ── HuggingFace Inference API ─────────────────────────────────────────────────

_HF_MODEL = "facebook/bart-large-mnli"
_HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{_HF_MODEL}"

# Candidate labels for zero-shot classification
_LABELS = ["entailment", "neutral", "contradiction"]

# Score contribution per verdict
_VERDICT_WEIGHTS = {"entailment": 1.0, "neutral": 0.4, "contradiction": 0.0}

# ── Claim extraction prompt (shared with RAG) ─────────────────────────────────

_EXTRACT_CLAIMS_PROMPT = (
    "Extract all distinct, non-redundant factual claims from the text. "
    "Each claim must be unique — do not repeat the same fact in different words. "
    'Return ONLY a JSON object: {"claims": ["claim 1", "claim 2", ...]}'
)


class NLIEntailmentDetector(BaseDetector):
    """Non-LLM hallucination signal via natural language inference."""

    name = "nli_entailment"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        hf_api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.llm = llm_client or LLMClient()
        key = hf_api_key or os.getenv("HF_API_KEY", "")
        self._hf_key: str | None = key if key else None
        self._http_client = http_client

    # ── public ───────────────────────────────────────────────────────────────

    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] | None = None,
    ) -> DetectorResult:
        if not self._hf_key:
            return DetectorResult(
                name=self.name,
                score=0.5,
                evidence=[
                    "HF_API_KEY not configured — NLI entailment check skipped.",
                    "Score defaulted to 0.5 (unverifiable).",
                    "Set HF_API_KEY in backend/.env to enable NLI grounding.",
                ],
                metadata={"configured": False},
            )

        # Step 1 — extract claims
        claims = await self._extract_claims(response)
        if not claims:
            return DetectorResult(
                name=self.name,
                score=0.7,
                evidence=["No factual claims extracted from response."],
                metadata={"claims_extracted": 0},
            )

        # Step 2 — classify each claim in parallel against the premise (context docs if available)
        if context_docs:
            premise_parts = []
            char_count = 0
            for doc in context_docs:
                doc_str = str(doc).strip()
                if not doc_str:
                    continue
                if char_count + len(doc_str) > 3000:
                    break
                premise_parts.append(doc_str)
                char_count += len(doc_str)
            premise = "\n".join(premise_parts) if premise_parts else query
        else:
            premise = query

        sem = asyncio.Semaphore(2)  # Limit concurrency to 2 to prevent Hugging Face rate limits/hangs

        async def sem_classify(p, h):
            async with sem:
                return await self._classify(p, h)

        tasks = [sem_classify(premise, claim) for claim in claims]
        verdicts: list[dict[str, Any]] = await asyncio.gather(*tasks)

        # Step 3 — compute score
        verified: list[str] = []
        unverified: list[str] = []
        evidence_lines: list[str] = []
        score_sum = 0.0

        for claim, verdict in zip(claims, verdicts):
            label = verdict.get("label", "neutral")
            conf = verdict.get("score", 0.0)
            error = verdict.get("error")
            contribution = _VERDICT_WEIGHTS.get(label, 0.4)
            score_sum += contribution

            if error:
                ev = f"⚠️ {claim} → Error: {error}"
            else:
                ev = f"{claim} → {label} ({conf:.0%} confidence)"
            evidence_lines.append(ev)

            if label == "entailment":
                verified.append(claim)
            elif label == "contradiction":
                unverified.append(claim)
            # neutral claims are excluded from both lists

        total = len(claims) or 1
        final_score = max(0.0, min(1.0, score_sum / total))

        return DetectorResult(
            name=self.name,
            score=final_score,
            verified_claims=verified,
            unverified_claims=unverified,
            evidence=evidence_lines,
            metadata={
                "model": _HF_MODEL,
                "claims_extracted": len(claims),
                "entailed": len(verified),
                "contradicted": len(unverified),
                "neutral": len(claims) - len(verified) - len(unverified),
            },
        )

    # ── internals ────────────────────────────────────────────────────────────

    async def _classify(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """Call HF zero-shot classification and return the top label + score."""
        template = f"This text states that: {hypothesis} — {{}}."
        payload = {
            "inputs": premise,
            "parameters": {
                "candidate_labels": _LABELS,
                "hypothesis_template": template,
            },
        }
        headers = {"Authorization": f"Bearer {self._hf_key}"}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._http_client:
                    resp = await self._http_client.post(_HF_API_URL, headers=headers, json=payload, timeout=10.0)
                    
                    if resp.status_code == 503:
                        # Model is loading — wait and retry
                        data = resp.json()
                        wait_time = data.get("estimated_time", 5.0)
                        await asyncio.sleep(min(wait_time, 10.0))
                        continue
                        
                    resp.raise_for_status()
                    data = resp.json()
                    break
                else:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        resp = await client.post(_HF_API_URL, headers=headers, json=payload, timeout=10.0)
                        
                        if resp.status_code == 503:
                            # Model is loading — wait and retry
                            data = resp.json()
                            wait_time = data.get("estimated_time", 5.0)
                            await asyncio.sleep(min(wait_time, 10.0))
                            continue
                            
                        resp.raise_for_status()
                        data = resp.json()
                        break
            except Exception as exc:
                if attempt == max_retries - 1:
                    return {"label": "neutral", "score": 0.0, "error": str(exc)}
                await asyncio.sleep(2.0)

        # HF Zero-Shot response: {"labels": ["entailment", ...], "scores": [0.92, ...]}
        if isinstance(data, dict) and "labels" in data and "scores" in data:
            labels = data["labels"]
            scores = data["scores"]
            if labels and scores:
                label = labels[0].lower()
                score = scores[0]
                # If entailment confidence is low, fall back to neutral to avoid false positives
                if label == "entailment" and score < 0.50:
                    label = "neutral"
                return {
                    "label": label,
                    "score": score,
                }

        # Fallback for standard classification format
        if isinstance(data, list) and data and isinstance(data[0], dict):
            top = data[0]
            label = top.get("label", "neutral").lower()
            score = top.get("score", 0.0)
            if label == "entailment" and score < 0.50:
                label = "neutral"
            return {
                "label": label,
                "score": score,
            }

        return {"label": "neutral", "score": 0.0, "error": "Unexpected API response format"}

    async def _extract_claims(self, text: str) -> list[str]:
        """Extract atomic factual claims from the response."""
        result = await asyncio.to_thread(
            self.llm.chat_json,
            [
                {"role": "system", "content": _EXTRACT_CLAIMS_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        return result.get("claims", [])
