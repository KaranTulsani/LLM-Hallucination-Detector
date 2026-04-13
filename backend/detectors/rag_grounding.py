"""
RAG Grounding Detector

Retrieves top-k documents from a lightweight in-memory vector store
and checks whether the LLM response is supported by those documents.

Uses fastembed for embeddings + numpy cosine similarity (no ChromaDB
needed — avoids the chroma-hnswlib C++ build requirement).

Two-step verification:
  1. Semantic retrieval — cosine similarity via fastembed.
  2. LLM verification — asks the LLM to classify each claim as
     supported / partially supported / contradicted by the retrieved docs.

If no knowledge base exists or yields no results, returns a penalty
score (0.5) to prevent artificially high scores for unverifiable claims.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from llm_client import LLMClient
from detectors.base import BaseDetector, DetectorResult

_VERIFY_PROMPT = """\
You are a claim verification assistant. You will be given a list of claims \
and a set of reference documents.

For each claim, determine if it is:
- "supported" — the documents contain evidence that confirms the claim
- "partially_supported" — the documents are related but don't fully confirm
- "contradicted" — the documents contain evidence that contradicts the claim
- "not_found" — the documents don't address this claim at all

Return ONLY a JSON object:
{
  "results": [
    {"claim": "...", "verdict": "supported|partially_supported|contradicted|not_found", "evidence_snippet": "..."},
    ...
  ]
}"""


class SimpleVectorStore:
    """Lightweight in-memory vector store using fastembed + numpy.

    Avoids the ChromaDB/hnswlib C++ compilation dependency.
    Perfectly adequate for knowledge bases up to ~10k documents.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._embedder = TextEmbedding(model_name=model_name)
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._ids: list[str] = []

    @property
    def count(self) -> int:
        return len(self._documents)

    def add(self, documents: list[str], ids: list[str] | None = None):
        """Add documents to the store."""
        new_embeddings = np.array(list(self._embedder.embed(documents)))

        if ids is None:
            start = len(self._ids)
            ids = [f"doc_{start + i}" for i in range(len(documents))]

        self._documents.extend(documents)
        self._ids.extend(ids)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

    def query(self, query_text: str, n_results: int = 5) -> dict:
        """Retrieve top-k most similar documents."""
        if self._embeddings is None or len(self._documents) == 0:
            return {"documents": [], "distances": [], "ids": []}

        query_emb = np.array(list(self._embedder.embed([query_text])))[0]

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1)
        norms = np.where(norms == 0, 1e-10, norms)
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            query_norm = 1e-10

        similarities = self._embeddings @ query_emb / (norms * query_norm)

        # Top-k indices
        k = min(n_results, len(self._documents))
        top_indices = np.argsort(similarities)[::-1][:k]

        return {
            "documents": [self._documents[i] for i in top_indices],
            "distances": [float(1 - similarities[i]) for i in top_indices],
            "ids": [self._ids[i] for i in top_indices],
        }

    def reset(self):
        """Clear all documents."""
        self._documents = []
        self._embeddings = None
        self._ids = []


class RAGGroundingDetector(BaseDetector):
    """Ground-truth verification against a knowledge base."""

    name = "rag_grounding"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        top_k: int = 5,
    ):
        self.llm = llm_client or LLMClient()
        self.top_k = top_k
        self._store = SimpleVectorStore()

    # ── public ──────────────────────────────────────────────────────

    @property
    def has_knowledge_base(self) -> bool:
        return self._store.count > 0

    async def score(
        self,
        query: str,
        response: str,
        context_docs: list[str] | None = None,
    ) -> DetectorResult:
        """Score groundedness against retrieved documents."""

        # If caller already supplies context docs, use those directly
        if context_docs:
            return await self._verify_against_docs(query, response, context_docs)

        # Otherwise, try to retrieve from the vector store
        if not self.has_knowledge_base:
            return DetectorResult(
                name=self.name,
                score=0.5,  # penalty for unverifiable
                evidence=[
                    "No knowledge base available — cannot verify claims.",
                    "Score defaulted to 0.5 (unverifiable penalty).",
                ],
                metadata={"has_knowledge_base": False},
            )

        # Retrieve relevant documents
        results = await asyncio.to_thread(
            self._store.query,
            query,
            n_results=self.top_k,
        )

        docs = results.get("documents", [])
        distances = results.get("distances", [])

        if not docs:
            return DetectorResult(
                name=self.name,
                score=0.5,
                evidence=["No relevant documents found in knowledge base."],
                metadata={"has_knowledge_base": True, "docs_retrieved": 0},
            )

        return await self._verify_against_docs(
            query, response, docs, distances=distances
        )

    # ── internals ───────────────────────────────────────────────────

    async def _verify_against_docs(
        self,
        query: str,
        response: str,
        docs: list[str],
        distances: list[float] | None = None,
    ) -> DetectorResult:
        """Ask the LLM to verify claims against reference documents."""

        # Step 1 — extract claims
        claims = await self._extract_claims(response)
        if not claims:
            return DetectorResult(
                name=self.name,
                score=0.7,  # no claims = somewhat trust it
                evidence=["No factual claims extracted from response."],
                metadata={"claims_extracted": 0},
            )

        # Step 2 — verify each claim against docs
        docs_text = "\n\n---\n\n".join(docs[: self.top_k])
        claims_text = "\n".join(f"- {c}" for c in claims)

        verify_msg = (
            f"**Claims to verify:**\n{claims_text}\n\n"
            f"**Reference documents:**\n{docs_text}"
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
                metadata={"raw": result.get("raw", "")},
            )

        verdicts = result.get("results", [])
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
            f'(evidence: {v.get("evidence_snippet", "N/A")})'
            for v in verdicts
        ]

        return DetectorResult(
            name=self.name,
            score=grounding_score,
            verified_claims=verified,
            unverified_claims=unverified,
            evidence=evidence_snippets,
            metadata={
                "docs_retrieved": len(docs),
                "claims_extracted": len(claims),
                "supported": len(supported),
                "partially_supported": len(partial),
                "contradicted": len(contradicted),
                "not_found": len(not_found),
                "distances": distances or [],
            },
        )

    async def _extract_claims(self, text: str) -> list[str]:
        """Use the LLM to extract atomic factual claims from text."""
        result = await asyncio.to_thread(
            self.llm.chat_json,
            [
                {
                    "role": "system",
                    "content": (
                        "Extract all distinct factual claims from the text. "
                        'Return ONLY a JSON object: {"claims": ["claim 1", "claim 2", ...]}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        return result.get("claims", [])

    # ── knowledge-base management ───────────────────────────────────

    def add_documents(self, documents: list[str], ids: list[str] | None = None):
        """Add documents to the vector store."""
        self._store.add(documents, ids)

    def reset_knowledge_base(self):
        """Clear the knowledge base."""
        self._store.reset()