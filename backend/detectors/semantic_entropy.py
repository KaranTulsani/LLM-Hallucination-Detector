"""
Semantic Entropy Detector (Self-Consistency Check)

Runs the same query N times with temperature > 0, computes pairwise
semantic similarity between responses using TF-IDF cosine similarity,
and derives a consistency score.

High consistency → model is confident → higher score
Low consistency  → model is guessing  → lower score

Note: consistency ≠ correctness, but inconsistency strongly signals
hallucination.
"""

from __future__ import annotations

import asyncio
import os
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import LLMClient
from detectors.base import BaseDetector, DetectorResult


class SemanticEntropyDetector(BaseDetector):
    """Detect hallucination via multi-sample semantic consistency."""

    name = "semantic_entropy"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        n_samples: int | None = None,
        temperature: float = 0.7,
    ):
        self.llm = llm_client or LLMClient()
        self.n_samples = n_samples or int(
            os.getenv("SELF_CONSISTENCY_SAMPLES", "5")
        )
        self.temperature = temperature
        # TF-IDF vectorizer: lightweight, CPU-only, no warmup needed
        self._vectorizer = TfidfVectorizer()

    # ── public ──────────────────────────────────────────────────────

    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] | None = None,
    ) -> DetectorResult:
        # Generate N alternative responses concurrently in thread pools
        tasks = [
            asyncio.to_thread(self.llm.generate_response, query, self.temperature)
            for _ in range(self.n_samples)
        ]
        samples = await asyncio.gather(*tasks)

        # Include the original response in the pool
        all_responses = [response] + samples

        # Compute TF-IDF cosine similarity between all response pairs
        similarities = await asyncio.to_thread(
            self._compute_similarities, all_responses
        )
        avg_similarity = float(np.mean(similarities)) if similarities else 0.5

        # Map similarity → score (higher similarity = more consistent)
        consistency_score = self._calibrate(avg_similarity)

        return DetectorResult(
            name=self.name,
            score=consistency_score,
            metadata={
                "n_samples": self.n_samples,
                "avg_pairwise_similarity": round(avg_similarity, 4),
                "min_pairwise_similarity": round(
                    float(np.min(similarities)) if similarities else 0.0, 4
                ),
                "max_pairwise_similarity": round(
                    float(np.max(similarities)) if similarities else 0.0, 4
                ),
            },
            evidence=[
                f"Generated {self.n_samples} alternative responses.",
                f"Average pairwise cosine similarity: {avg_similarity:.3f}",
            ],
        )

    # ── helpers ─────────────────────────────────────────────────────

    def _compute_similarities(self, texts: list[str]) -> list[float]:
        """Compute pairwise TF-IDF cosine similarity for all response pairs."""
        if len(texts) < 2:
            return [1.0]
        try:
            tfidf_matrix = self._vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception:
            return [0.5]
        sims: list[float] = []
        for i, j in combinations(range(len(texts)), 2):
            sims.append(float(sim_matrix[i, j]))
        return sims

    @staticmethod
    def _pairwise_cosine(matrix: np.ndarray) -> list[float]:
        """Retained for reference only (not called in current implementation)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = matrix / norms
        sims: list[float] = []
        for i, j in combinations(range(len(matrix)), 2):
            sims.append(float(np.dot(normed[i], normed[j])))
        return sims

    @staticmethod
    def _calibrate(avg_sim: float) -> float:
        """Map average TF-IDF cosine similarity to a 0–1 confidence score.

        Thresholds:
          sim >= 0.92 → full confidence (1.0)
          sim <= 0.60 → no confidence  (0.0)
          Linear interpolation in between.

        Why 0.92 not 1.0?  Slight phrasing variations in highly consistent
        answers (e.g. synonyms) naturally prevent a perfect cosine similarity.
        A value like 0.98 is already "perfectly consistent."
        """
        LOW, HIGH = 0.60, 0.92
        if avg_sim >= HIGH:
            return 1.0
        if avg_sim <= LOW:
            return 0.0
        return (avg_sim - LOW) / (HIGH - LOW)