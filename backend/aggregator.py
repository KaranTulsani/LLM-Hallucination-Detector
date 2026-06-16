"""
Score Aggregator

Combines sub-scores from all detectors into a single 0–100 score
using configurable weights and applies bonus penalty signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from detectors.base import DetectorResult

# Claims with cosine similarity above this threshold are considered
# semantically equivalent (e.g. same fact worded differently by different
# detectors).  0.55 is deliberately conservative—tight enough not to
# merge genuinely distinct claims, loose enough to catch reformulations.
_SEMANTIC_SIM_THRESHOLD = 0.55


def _dedup_claim_list(claims: list[str]) -> list[str]:
    """Deduplicate claims by normalised (lowercased, stripped, no trailing period) text.

    Preserves the original casing of the first occurrence.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for c in claims:
        key = c.strip().rstrip(".").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _remove_semantic_duplicates(
    unverified: list[str],
    verified: list[str],
    threshold: float = _SEMANTIC_SIM_THRESHOLD,
) -> list[str]:
    """Remove unverified claims that are semantically covered by a verified claim.

    Uses TF-IDF cosine similarity to catch differently-worded variants of
    the same factual claim (e.g. "The IBM Simon was released in 1994 and
    had email" vs "The IBM Simon had email, fax, and phone capabilities").

    Returns the filtered list of genuinely-unverified claims.
    """
    if not unverified or not verified:
        return unverified

    try:
        vectorizer = TfidfVectorizer()
        # Fit on all claims, then compute cross-similarity
        all_texts = verified + unverified
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        n_verified = len(verified)
        verified_matrix = tfidf_matrix[:n_verified]
        unverified_matrix = tfidf_matrix[n_verified:]

        sim_matrix = cosine_similarity(unverified_matrix, verified_matrix)

        filtered: list[str] = []
        for i, claim in enumerate(unverified):
            max_sim = float(np.max(sim_matrix[i]))
            if max_sim < threshold:
                # Not covered by any verified claim → keep as unverified
                filtered.append(claim)
        return filtered
    except Exception:
        # Fallback: if TF-IDF fails for any reason, return original list
        return unverified


@dataclass
class AggregatedScore:
    """Final output of the scoring pipeline."""

    final_score: float                     # 0–100
    sub_scores: dict[str, float]           # each detector's 0–1 score
    verified_claims: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    is_trustworthy: bool = True
    threshold_used: float = 65.0
    citations: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 1),
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "evidence": self.evidence,
            "penalties": self.penalties,
            "is_trustworthy": self.is_trustworthy,
            "threshold_used": self.threshold_used,
            "citations": self.citations,
        }


class ScoreAggregator:
    """Weighted combination of detector scores with penalty signals."""

    # Default weights from the spec
    DEFAULT_WEIGHTS = {
        "rag_grounding": 0.40,
        "llm_judge": 0.25,        # repurposed from nli_entailment weight
        "semantic_entropy": 0.20,
        "nli_entailment": 0.15,   # reserved for future NLI detector
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold: float = 65.0,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold

    def aggregate(
        self,
        results: list[DetectorResult],
        response_text: str = "",
    ) -> AggregatedScore:
        """Combine detector results into a final 0–100 score."""

        sub_scores: dict[str, float] = {}
        all_verified: list[str] = []
        all_unverified: list[str] = []
        all_evidence: list[str] = []
        penalties: list[str] = []

        # Collect sub-scores
        citations: dict[str, list[dict[str, str]]] = {}
        for r in results:
            sub_scores[r.name] = r.score
            all_verified.extend(r.verified_claims)
            all_unverified.extend(r.unverified_claims)
            all_evidence.extend(r.evidence)
            if r.metadata and "citations" in r.metadata:
                citations.update(r.metadata["citations"])

        # Weighted sum — only include detectors that actually ran
        active_weights = {
            k: v for k, v in self.weights.items() if k in sub_scores
        }

        if not active_weights:
            return AggregatedScore(
                final_score=50.0,
                sub_scores=sub_scores,
                evidence=["No detectors ran — defaulting to 50."],
                is_trustworthy=False,
                threshold_used=self.threshold,
            )

        # Re-normalise weights so active ones sum to 1.0
        weight_sum = sum(active_weights.values())
        normed = {k: v / weight_sum for k, v in active_weights.items()}

        weighted_score = sum(
            normed.get(name, 0.0) * score
            for name, score in sub_scores.items()
        )

        final = weighted_score * 100.0

        # ── Penalty signals ─────────────────────────────────────────

        # 1. Overconfidence penalty — no hedging language on long responses
        if len(response_text) > 200:
            hedging = re.search(
                r"\b(may|might|could|possibly|perhaps|according to|it is "
                r"possible|some suggest|it appears)\b",
                response_text,
                re.IGNORECASE,
            )
            if not hedging:
                final -= 5
                penalties.append(
                    "Overconfidence: no hedging language detected (-5)"
                )

        # 2. Response length mismatch
        if len(response_text) > 1500 and len(response_text.split("?")) <= 2:
            # Very long answer that's not a list of questions
            final -= 3
            penalties.append(
                "Length mismatch: unusually long response (-3)"
            )

        # 3. Citation check penalty (simplified — URL pattern detection)
        urls = re.findall(r"https?://[^\s)\"'>]+", response_text)
        if urls:
            # We flag them; real HTTP checking happens asynchronously
            # in the corrector or as a future enhancement
            penalties.append(
                f"Contains {len(urls)} URL(s) — citation validity not yet verified"
            )

        final = max(0.0, min(100.0, final))

        # ── Deduplicate & resolve conflicting claims ─────────────
        # Different detectors may classify the same claim differently.
        # We prioritize strong detectors ('rag_grounding', 'llm_judge') over weaker
        # zero-shot NLI models, preventing false-positive NLI entailments from overriding
        # actual contradictions/unverified flags.
        verified_by_strong = []
        unverified_by_strong = []

        for r in results:
            if r.name in ("rag_grounding", "llm_judge"):
                verified_by_strong.extend(r.verified_claims)
                unverified_by_strong.extend(r.unverified_claims)

        deduped_strong_verified = _dedup_claim_list(verified_by_strong)
        deduped_strong_unverified = _dedup_claim_list(unverified_by_strong)

        if verified_by_strong or unverified_by_strong:
            # If strong detectors are active, their unverified verdicts override verified ones
            strong_unverified_keys = {c.strip().rstrip(".").strip().lower() for c in deduped_strong_unverified}
            final_verified = [c for c in deduped_strong_verified if c.strip().rstrip(".").strip().lower() not in strong_unverified_keys]
            final_unverified = deduped_strong_unverified

            # Include weaker detector claims only if not already covered by strong detectors
            seen_keys = {c.strip().rstrip(".").strip().lower() for c in (final_verified + final_unverified)}
            for r in results:
                if r.name not in ("rag_grounding", "llm_judge"):
                    for c in r.verified_claims:
                        if c.strip().rstrip(".").strip().lower() not in seen_keys:
                            final_verified.append(c)
                            seen_keys.add(c.strip().rstrip(".").strip().lower())
                    for c in r.unverified_claims:
                        if c.strip().rstrip(".").strip().lower() not in seen_keys:
                            final_unverified.append(c)
                            seen_keys.add(c.strip().rstrip(".").strip().lower())
        else:
            # Fallback to simple union if no strong detectors ran
            final_verified = _dedup_claim_list(all_verified)
            final_unverified = _dedup_claim_list(all_unverified)
            verified_keys = {c.strip().rstrip(".").strip().lower() for c in final_verified}
            final_unverified = [c for c in final_unverified if c.strip().rstrip(".").strip().lower() not in verified_keys]

        # Remove unverified claims that are *semantically* covered by a verified claim
        final_unverified = _remove_semantic_duplicates(
            final_unverified, final_verified
        )

        return AggregatedScore(
            final_score=final,
            sub_scores=sub_scores,
            verified_claims=final_verified,
            unverified_claims=final_unverified,
            evidence=all_evidence,
            penalties=penalties,
            is_trustworthy=final >= self.threshold,
            threshold_used=self.threshold,
            citations=citations,
        )