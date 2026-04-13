"""
Abstract base class for all hallucination detectors.

Every detector implements a single interface:
    score(query, response, context_docs?) → DetectorResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectorResult:
    """Standardised output from every detector."""

    name: str                              # detector identifier
    score: float                           # 0.0 (hallucinated) → 1.0 (trustworthy)
    verified_claims: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


class BaseDetector(ABC):
    """Interface every detector must satisfy."""

    name: str = "base"

    @abstractmethod
    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] | None = None,
    ) -> DetectorResult:
        """Analyse *response* for hallucination signals.

        Parameters
        ----------
        query : str
            The original user query.
        response : str
            The LLM-generated response to evaluate.
        context_docs : list[str] | None
            Optional reference documents the response should be grounded in.

        Returns
        -------
        DetectorResult
            Score in [0, 1] and supporting evidence.
        """
        ...
