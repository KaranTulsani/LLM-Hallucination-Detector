"""
LLM-as-Judge Detector

Sends the original query + response to a separate LLM call with a
structured fact-checking prompt.  The judge returns:
  - verified_claims   : claims it considers accurate
  - unverified_claims : claims it considers hallucinated / unsupported
  - confidence_score  : 0–1 overall trustworthiness rating

This is cheap, fast, and surprisingly effective when combined with
the other detectors.  On its own it's unreliable because the judge
LLM can also hallucinate — use it as one signal, not the verdict.
"""

from __future__ import annotations

import asyncio

from llm_client import LLMClient
from detectors.base import BaseDetector, DetectorResult

_JUDGE_SYSTEM_PROMPT = """\
You are a rigorous fact-checking AI. Your job is to evaluate the factual \
accuracy and groundedness of an AI-generated response.

Analyse the response carefully and return ONLY a JSON object with this \
exact schema — no markdown fences, no extra text:

{
  "verified_claims": ["claim that appears factually correct", ...],
  "unverified_claims": ["claim that appears hallucinated, fabricated, or unsupported", ...],
  "fabricated_citations": ["any URLs, paper titles, or references that appear invented", ...],
  "overconfidence_flags": ["statements presented with certainty on contested/uncertain topics", ...],
  "confidence_score": <float 0-1, where 1 = fully trustworthy>
}

Rules:
1. A claim is "unverified" if you cannot confirm it OR if it contradicts \
   well-established facts.
2. Check for fabricated citations: invented paper titles, fake URLs, \
   non-existent author names.
3. Flag overconfidence: phrases like "it is well established" or \
   "studies show" without specific sources on contested topics.
4. Be conservative — when in doubt, mark as unverified.
5. Return ONLY valid JSON. No other text."""


class LLMJudgeDetector(BaseDetector):
    """Use an LLM to fact-check another LLM's response."""

    name = "llm_judge"

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] | None = None,
    ) -> DetectorResult:
        user_msg = (
            f"**User query:** {query}\n\n"
            f"**AI response to evaluate:**\n{response}"
        )

        if context_docs:
            docs_text = "\n---\n".join(context_docs[:5])
            user_msg += (
                f"\n\n**Reference documents (ground truth):**\n{docs_text}"
            )

        result = await asyncio.to_thread(
            self.llm.chat_json,
            [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )

        if result.get("parse_error"):
            # If JSON extraction failed, return a conservative mid-range score
            return DetectorResult(
                name=self.name,
                score=0.5,
                evidence=["Judge response could not be parsed as JSON."],
                metadata={"raw_response": result.get("raw", "")},
            )

        verified = result.get("verified_claims", [])
        unverified = result.get("unverified_claims", [])
        fabricated = result.get("fabricated_citations", [])
        overconfidence = result.get("overconfidence_flags", [])
        raw_confidence = float(result.get("confidence_score", 0.5))

        # Compute score from claim ratio
        total = len(verified) + len(unverified)
        claim_ratio = len(verified) / total if total > 0 else 0.5

        # Blend the judge's own confidence with the claim ratio
        blended_score = 0.6 * claim_ratio + 0.4 * raw_confidence

        # Penalties
        if fabricated:
            blended_score -= 0.15 * min(len(fabricated), 3)
        if overconfidence:
            blended_score -= 0.05 * min(len(overconfidence), 3)

        blended_score = max(0.0, min(1.0, blended_score))

        return DetectorResult(
            name=self.name,
            score=blended_score,
            verified_claims=verified,
            unverified_claims=unverified,
            evidence=[
                f"Judge confidence: {raw_confidence:.2f}",
                f"Verified: {len(verified)}, Unverified: {len(unverified)}",
                *(
                    [f"Fabricated citations detected: {fabricated}"]
                    if fabricated else []
                ),
                *(
                    [f"Overconfidence flags: {overconfidence}"]
                    if overconfidence else []
                ),
            ],
            metadata={
                "raw_confidence": raw_confidence,
                "claim_ratio": round(claim_ratio, 4),
                "fabricated_citations": fabricated,
                "overconfidence_flags": overconfidence,
            },
        )