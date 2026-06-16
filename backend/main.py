"""
Hallucination Detector — FastAPI Application

Main entrypoints:
  POST /api/detect       — score a query+response pair
  POST /api/correct      — fix a hallucinated response
  POST /api/chat         — generate + auto-detect in one call
  POST /api/kb/add       — DEPRECATED (web search replaced static KB)
  DELETE /api/kb/reset   — DEPRECATED (web search replaced static KB)
  GET  /api/health       — health check
"""

from __future__ import annotations

import asyncio
import json
import os
import httpx
import cache_manager
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aggregator import ScoreAggregator
from corrector import Corrector
from detectors import LLMJudgeDetector, NLIEntailmentDetector, RAGGroundingDetector, SemanticEntropyDetector
from llm_client import LLMClient

# Load .env relative to this file so it works regardless of CWD
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# ── Shared instances (created once at startup) ──────────────────────

llm_client: LLMClient | None = None
judge_llm_client: LLMClient | None = None
semantic_detector: SemanticEntropyDetector | None = None
judge_detector: LLMJudgeDetector | None = None
rag_detector: RAGGroundingDetector | None = None
nli_detector: NLIEntailmentDetector | None = None
aggregator: ScoreAggregator | None = None
corrector: Corrector | None = None
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy objects once, share across requests."""
    global llm_client, judge_llm_client, semantic_detector, judge_detector
    global rag_detector, nli_detector, aggregator, corrector, http_client

    threshold = float(os.getenv("DETECTION_THRESHOLD", "65"))
    judge_model = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")

    http_client = httpx.AsyncClient(timeout=30.0)
    llm_client = LLMClient()  # fast model for generation (llama-3.1-8b-instant)
    judge_llm_client = LLMClient(model=judge_model)  # stronger model for judging
    semantic_detector = SemanticEntropyDetector(llm_client=llm_client)
    judge_detector = LLMJudgeDetector(llm_client=judge_llm_client)
    rag_detector = RAGGroundingDetector(llm_client=llm_client, http_client=http_client)
    nli_detector = NLIEntailmentDetector(llm_client=llm_client, http_client=http_client)
    aggregator = ScoreAggregator(threshold=threshold)
    corrector = Corrector(llm_client=llm_client)

    yield  # app runs

    # cleanup
    await http_client.aclose()


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
        description="Which detectors to run. Default: all. Options: semantic_entropy, llm_judge, rag_grounding, nli_entailment",
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
    history: list[dict[str, str]] | None = Field(
        default=None, description="Previous messages in chat session"
    )


class KBAddRequest(BaseModel):
    """Deprecated: retained for backward-compatibility only."""
    documents: list[str]
    ids: list[str] | None = None


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": llm_client.model if llm_client else "unknown",
        "judge_model": judge_llm_client.model if judge_llm_client else "unknown",
    }


@app.post("/api/detect")
async def detect(req: DetectRequest):
    """Run hallucination detection on a query+response pair."""

    # Check cache first
    cached = cache_manager.get_cached_detect(req.query, req.response)
    if cached:
        return cached

    selected = req.detectors or ["semantic_entropy", "llm_judge", "rag_grounding", "nli_entailment"]
    
    context_docs = req.context_docs
    if (
        not context_docs
        and "rag_grounding" in selected
        and rag_detector
        and rag_detector.has_knowledge_base
    ):
        claims = await rag_detector._extract_claims(req.response)
        claims = [c.strip() for c in claims if c.strip()]
        if claims:
            seen_claims = set()
            unique_claims = []
            for c in claims:
                key = c.lower()
                if key not in seen_claims:
                    seen_claims.add(key)
                    unique_claims.append(c)
            search_tasks = [rag_detector._retriever.search(c) for c in unique_claims]
            per_claim_snippets = await asyncio.gather(*search_tasks)
            seen_snippets = set()
            docs = []
            for snippets in per_claim_snippets:
                for s in snippets[:2]:
                    if s["text"] not in seen_snippets and not s["text"].startswith("[Web search failed:"):
                        seen_snippets.add(s["text"])
                        docs.append(s["text"])
            if docs:
                context_docs = docs

    tasks = []
    if "semantic_entropy" in selected and semantic_detector:
        tasks.append(semantic_detector.score(req.query, req.response, context_docs))
    if "llm_judge" in selected and judge_detector:
        tasks.append(judge_detector.score(req.query, req.response, context_docs))
    if "rag_grounding" in selected and rag_detector:
        tasks.append(rag_detector.score(req.query, req.response, context_docs))
    if "nli_entailment" in selected and nli_detector:
        tasks.append(nli_detector.score(req.query, req.response, context_docs))

    results = await asyncio.gather(*tasks)
    agg = aggregator.aggregate(list(results), req.response)

    result_data = {
        "score": agg.to_dict(),
        "detectors": [r.to_dict() for r in results],
    }
    cache_manager.save_cached_detect(req.query, req.response, result_data)
    return result_data


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

    # Check cache first
    cached = cache_manager.get_cached_chat(req.query)
    if cached:
        return cached

    # Step 1 — Generate response (including history context if available)
    messages = []
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.query})

    generated = await asyncio.to_thread(
        llm_client.chat, messages
    )

    # Step 2 — Run detection
    context_docs = req.context_docs
    if (
        not context_docs
        and rag_detector
        and rag_detector.has_knowledge_base
    ):
        claims = await rag_detector._extract_claims(generated)
        claims = [c.strip() for c in claims if c.strip()]
        if claims:
            seen_claims = set()
            unique_claims = []
            for c in claims:
                key = c.lower()
                if key not in seen_claims:
                    seen_claims.add(key)
                    unique_claims.append(c)
            search_tasks = [rag_detector._retriever.search(c) for c in unique_claims]
            per_claim_snippets = await asyncio.gather(*search_tasks)
            seen_snippets = set()
            docs = []
            for snippets in per_claim_snippets:
                for s in snippets[:2]:
                    if s["text"] not in seen_snippets and not s["text"].startswith("[Web search failed:"):
                        seen_snippets.add(s["text"])
                        docs.append(s["text"])
            if docs:
                context_docs = docs

    tasks = [
        semantic_detector.score(req.query, generated, context_docs),
        judge_detector.score(req.query, generated, context_docs),
        rag_detector.score(req.query, generated, context_docs),
        nli_detector.score(req.query, generated, context_docs),
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

    # Step 3 — Auto-fix if requested and below threshold (Verify-and-Correct Loop)
    if req.auto_fix and not agg.is_trustworthy:
        current_response = generated
        current_agg = agg
        
        for round_num in range(2):
            fixed = await corrector.constrained_reprompt(
                req.query, current_response, current_agg.unverified_claims
            )
            
            # Re-run detectors on the fixed response
            new_tasks = [
                semantic_detector.score(req.query, fixed, context_docs),
                judge_detector.score(req.query, fixed, context_docs),
                rag_detector.score(req.query, fixed, context_docs),
                nli_detector.score(req.query, fixed, context_docs),
            ]
            new_results = await asyncio.gather(*new_tasks)
            new_agg = aggregator.aggregate(list(new_results), fixed)
            
            if new_agg.is_trustworthy or round_num == 1:
                response_data["was_corrected"] = True
                response_data["corrected_response"] = fixed
                response_data["score"] = new_agg.to_dict()
                response_data["detectors"] = [r.to_dict() for r in new_results]
                break
            else:
                current_response = fixed
                current_agg = new_agg

    cache_manager.save_cached_chat(req.query, response_data)
    return response_data


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Generate response stream, then detect hallucination."""
    
    # Check cache first
    cached = cache_manager.get_cached_chat(req.query)
    if cached:
        async def cached_stream_generator():
            text = cached.get("response", "")
            # Chunk the response into ~4 character tokens and stream them with 10ms typing delay
            for i in range(0, len(text), 4):
                yield f"data: {json.dumps({'chunk': text[i:i+4]})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'result': cached})}\n\n"
        return StreamingResponse(cached_stream_generator(), media_type="text/event-stream")

    q = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def producer(l):
        full_text = ""
        try:
            # Build history list
            messages = []
            if req.history:
                messages.extend(req.history)
            messages.append({"role": "user", "content": req.query})

            for chunk in llm_client.chat_stream(messages):
                asyncio.run_coroutine_threadsafe(q.put({"chunk": chunk}), l)
                full_text += chunk
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put({"error": str(e)}), l)
            return

        asyncio.run_coroutine_threadsafe(q.put({"done": True, "full_text": full_text}), l)

    loop.run_in_executor(None, producer, loop)

    async def stream_generator():
        while True:
            msg = await q.get()
            if "error" in msg:
                yield f"data: {json.dumps({'error': msg['error']})}\n\n"
                break
            elif "chunk" in msg:
                yield f"data: {json.dumps({'chunk': msg['chunk']})}\n\n"
            elif "done" in msg:
                full_text = msg["full_text"]
                
                context_docs = req.context_docs
                if (
                    not context_docs
                    and rag_detector
                    and rag_detector.has_knowledge_base
                ):
                    claims = await rag_detector._extract_claims(full_text)
                    claims = [c.strip() for c in claims if c.strip()]
                    if claims:
                        seen_claims = set()
                        unique_claims = []
                        for c in claims:
                            key = c.lower()
                            if key not in seen_claims:
                                seen_claims.add(key)
                                unique_claims.append(c)
                        search_tasks = [rag_detector._retriever.search(c) for c in unique_claims]
                        per_claim_snippets = await asyncio.gather(*search_tasks)
                        seen_snippets = set()
                        docs = []
                        for snippets in per_claim_snippets:
                            for s in snippets[:2]:  # take top 2 snippets per claim to prevent cutoff
                                if s["text"] not in seen_snippets and not s["text"].startswith("[Web search failed:"):
                                    seen_snippets.add(s["text"])
                                    docs.append(s["text"])
                        if docs:
                            context_docs = docs

                tasks = []
                if semantic_detector: tasks.append(semantic_detector.score(req.query, full_text, context_docs))
                if judge_detector: tasks.append(judge_detector.score(req.query, full_text, context_docs))
                if rag_detector: tasks.append(rag_detector.score(req.query, full_text, context_docs))
                if nli_detector: tasks.append(nli_detector.score(req.query, full_text, context_docs))
                
                results = await asyncio.gather(*tasks)
                agg = aggregator.aggregate(list(results), full_text)
                
                final_data = {
                    "query": req.query,
                    "response": full_text,
                    "score": agg.to_dict(),
                    "detectors": [r.to_dict() for r in results],
                    "was_corrected": False,
                    "corrected_response": None,
                }
                
                if req.auto_fix and not agg.is_trustworthy:
                    current_response = full_text
                    current_agg = agg
                    
                    for round_num in range(2):
                        fixed = await corrector.constrained_reprompt(
                            req.query, current_response, current_agg.unverified_claims
                        )
                        
                        # Re-run detectors on the fixed response
                        new_tasks = []
                        if semantic_detector: new_tasks.append(semantic_detector.score(req.query, fixed, context_docs))
                        if judge_detector: new_tasks.append(judge_detector.score(req.query, fixed, context_docs))
                        if rag_detector: new_tasks.append(rag_detector.score(req.query, fixed, context_docs))
                        if nli_detector: new_tasks.append(nli_detector.score(req.query, fixed, context_docs))
                        
                        new_results = await asyncio.gather(*new_tasks)
                        new_agg = aggregator.aggregate(list(new_results), fixed)
                        
                        if new_agg.is_trustworthy or round_num == 1:
                            final_data["was_corrected"] = True
                            final_data["corrected_response"] = fixed
                            final_data["score"] = new_agg.to_dict()
                            final_data["detectors"] = [r.to_dict() for r in new_results]
                            break
                        else:
                            current_response = fixed
                            current_agg = new_agg
                    
                cache_manager.save_cached_chat(req.query, final_data)
                yield f"data: {json.dumps({'result': final_data})}\n\n"
                break

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.post("/api/kb/add")
async def kb_add(req: KBAddRequest):
    """
    DEPRECATED — Web search replaced static KB.
    Kept for backward-compatibility; does nothing.
    """
    return {
        "status": "deprecated",
        "message": "Static knowledge base is no longer used. RAG grounding now uses live web search via Serper.dev.",
    }


@app.delete("/api/kb/reset")
async def kb_reset():
    """
    DEPRECATED — Web search replaced static KB.
    Kept for backward-compatibility; does nothing.
    """
    return {
        "status": "deprecated",
        "message": "Static knowledge base is no longer used. RAG grounding now uses live web search via Serper.dev.",
    }