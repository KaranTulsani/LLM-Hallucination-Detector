"""
Score Aggregator

Combines sub-scores from all detectors into a single 0–100 score
using configurable weights and applies bonus penalty signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from detectors.base import DetectorResult


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
        for r in results:
            sub_scores[r.name] = r.score
            all_verified.extend(r.verified_claims)
            all_unverified.extend(r.unverified_claims)
            all_evidence.extend(r.evidence)

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

        return AggregatedScore(
            final_score=final,
            sub_scores=sub_scores,
            verified_claims=list(set(all_verified)),
            unverified_claims=list(set(all_unverified)),
            evidence=all_evidence,
            penalties=penalties,
            is_trustworthy=final >= self.threshold,
            threshold_used=self.threshold,
        )