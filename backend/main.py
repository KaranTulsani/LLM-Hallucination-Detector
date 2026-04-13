"""
Hallucination Detector — FastAPI Application

Main entrypoints:
  POST /api/detect       — score a query+response pair
  POST /api/correct      — fix a hallucinated response
  POST /api/chat         — generate + auto-detect in one call
  POST /api/kb/add       — add documents to the knowledge base
  DELETE /api/kb/reset   — clear the knowledge base
  GET  /api/health       — health check
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aggregator import ScoreAggregator
from corrector import Corrector
from detectors import LLMJudgeDetector, RAGGroundingDetector, SemanticEntropyDetector
from llm_client import LLMClient

# Load .env relative to this file so it works regardless of CWD
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# ── Shared instances (created once at startup) ──────────────────────

llm_client: LLMClient | None = None
semantic_detector: SemanticEntropyDetector | None = None
judge_detector: LLMJudgeDetector | None = None
rag_detector: RAGGroundingDetector | None = None
aggregator: ScoreAggregator | None = None
corrector: Corrector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy objects once, share across requests."""
    global llm_client, semantic_detector, judge_detector
    global rag_detector, aggregator, corrector

    threshold = float(os.getenv("DETECTION_THRESHOLD", "65"))

    llm_client = LLMClient()
    semantic_detector = SemanticEntropyDetector(llm_client=llm_client)
    judge_detector = LLMJudgeDetector(llm_client=llm_client)
    rag_detector = RAGGroundingDetector(llm_client=llm_client)
    aggregator = ScoreAggregator(threshold=threshold)
    corrector = Corrector(llm_client=llm_client)

    yield  # app runs

    # cleanup (nothing heavy to tear down)


app = FastAPI(
    title="Hallucination Detector",
    version="1.0.0",
    description="Middleware that scores LLM responses for trustworthiness and optionally fixes hallucinations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ──────────────────────────────────────


class DetectRequest(BaseModel):
    query: str
    response: str
    context_docs: list[str] | None = None
    detectors: list[str] | None = Field(
        default=None,
        description="Which detectors to run. Default: all. Options: semantic_entropy, llm_judge, rag_grounding",
    )


class CorrectRequest(BaseModel):
    query: str
    response: str
    unverified_claims: list[str] = Field(default_factory=list)
    context_docs: list[str] | None = None
    strategy: str = Field(
        default="constrained",
        description="Correction strategy: constrained | rag | critic_loop",
    )


class ChatRequest(BaseModel):
    query: str
    context_docs: list[str] | None = None
    auto_fix: bool = Field(
        default=False, description="Automatically fix if below threshold"
    )


class KBAddRequest(BaseModel):
    documents: list[str]
    ids: list[str] | None = None


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": llm_client.model if llm_client else "unknown"}


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Run hallucination detection on a query+response pair."""

    selected = req.detectors or ["semantic_entropy", "llm_judge", "rag_grounding"]

    tasks = []
    if "semantic_entropy" in selected and semantic_detector:
        tasks.append(semantic_detector.score(req.query, req.response, req.context_docs))
    if "llm_judge" in selected and judge_detector:
        tasks.append(judge_detector.score(req.query, req.response, req.context_docs))
    if "rag_grounding" in selected and rag_detector:
        tasks.append(rag_detector.score(req.query, req.response, req.context_docs))

    results = await asyncio.gather(*tasks)
    agg = aggregator.aggregate(list(results), req.response)

    return {
        "score": agg.to_dict(),
        "detectors": [r.to_dict() for r in results],
    }


@app.post("/api/correct")
async def correct(req: CorrectRequest):
    """Fix a hallucinated response using the specified strategy."""

    if req.strategy == "rag" and req.context_docs:
        fixed = await corrector.rag_grounded_regeneration(
            req.query, req.context_docs
        )
    elif req.strategy == "critic_loop":
        fixed = await corrector.critic_generator_loop(
            req.query, req.response, req.context_docs
        )
    else:
        fixed = await corrector.constrained_reprompt(
            req.query, req.response, req.unverified_claims
        )

    return {"original_response": req.response, "corrected_response": fixed}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Generate a response, detect hallucination, and optionally fix it."""

    # Step 1 — Generate response
    generated = await asyncio.to_thread(
        llm_client.generate_response, req.query
    )

    # Step 2 — Run detection
    tasks = [
        semantic_detector.score(req.query, generated, req.context_docs),
        judge_detector.score(req.query, generated, req.context_docs),
        rag_detector.score(req.query, generated, req.context_docs),
    ]
    results = await asyncio.gather(*tasks)
    agg = aggregator.aggregate(list(results), generated)

    response_data = {
        "query": req.query,
        "response": generated,
        "score": agg.to_dict(),
        "detectors": [r.to_dict() for r in results],
        "was_corrected": False,
        "corrected_response": None,
    }

    # Step 3 — Auto-fix if requested and below threshold
    if req.auto_fix and not agg.is_trustworthy:
        fixed = await corrector.constrained_reprompt(
            req.query, generated, agg.unverified_claims
        )
        response_data["was_corrected"] = True
        response_data["corrected_response"] = fixed

    return response_data


@app.post("/api/kb/add")
async def kb_add(req: KBAddRequest):
    """Add documents to the in-memory vector store."""
    rag_detector.add_documents(req.documents, req.ids)
    return {
        "status": "ok",
        "documents_added": len(req.documents),
    }


@app.delete("/api/kb/reset")
async def kb_reset():
    """Clear the knowledge base."""
    rag_detector.reset_knowledge_base()
    return {"status": "ok", "message": "Knowledge base cleared."}