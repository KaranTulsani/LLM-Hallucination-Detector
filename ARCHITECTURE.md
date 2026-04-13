# Architecture

## System Overview

```
┌──────────────┐     ┌─────────────────────────────────────────────────────────┐
│   Frontend   │────▶│                    Backend (FastAPI)                     │
│  React + Vite│◀────│                                                         │
└──────────────┘     │  ┌─────────────────────────────────────────────────┐    │
                     │  │           Detection Pipeline                     │    │
                     │  │                                                  │    │
                     │  │  ┌─────────────┐  ┌───────────┐  ┌───────────┐  │    │
                     │  │  │  Semantic    │  │   LLM     │  │   RAG     │  │    │
                     │  │  │  Entropy     │  │   Judge   │  │  Ground   │  │    │
                     │  │  │  Detector    │  │  Detector │  │  Detector │  │    │
                     │  │  └──────┬──────┘  └─────┬─────┘  └─────┬─────┘  │    │
                     │  │         │               │               │        │    │
                     │  │         └───────────┬───┘───────────────┘        │    │
                     │  │                     ▼                            │    │
                     │  │             ┌──────────────┐                     │    │
                     │  │             │  Aggregator   │                     │    │
                     │  │             │  (Weighted)   │                     │    │
                     │  │             └──────┬───────┘                     │    │
                     │  └────────────────────┼────────────────────────────┘    │
                     │                       ▼                                 │
                     │              ┌──────────────┐                           │
                     │              │  Threshold    │                           │
                     │              │    Gate       │                           │
                     │              └──────┬───────┘                           │
                     │                     │                                    │
                     │          ┌──────────┴──────────┐                        │
                     │          ▼                      ▼                        │
                     │   ┌─────────────┐     ┌──────────────┐                  │
                     │   │  ✅ Deliver  │     │  Correction   │                  │
                     │   │  Response   │     │  Pipeline     │                  │
                     │   └─────────────┘     └──────────────┘                  │
                     └─────────────────────────────────────────────────────────┘
                                              │
                              ┌────────────────┼───────────────┐
                              ▼                ▼               ▼
                        ┌──────────┐   ┌────────────┐  ┌────────────┐
                        │  Groq    │   │  ChromaDB   │  │  fastembed  │
                        │  (LLM)   │   │  (Vector)   │  │  (Embed)    │
                        └──────────┘   └────────────┘  └────────────┘
```

## Data Flow

### 1. Request Entry

```
User → POST /api/chat → { query }
```

The backend generates a response using the LLM client, then pipes both query + response through the detection pipeline.

### 2. Parallel Detection

All three detectors execute simultaneously via `asyncio.gather()`:

- **SemanticEntropyDetector**: Generates N independent responses, embeds them with fastembed (BAAI/bge-small-en-v1.5), computes pairwise cosine similarity, and maps average similarity to a 0–1 confidence score.

- **LLMJudgeDetector**: Sends query+response to a structured fact-checking prompt. The judge returns verified/unverified claims, fabricated citations, and overconfidence flags. Score = blend of claim ratio and judge confidence, with penalties.

- **RAGGroundingDetector**: Extracts atomic claims from the response, retrieves top-k documents from ChromaDB, and has the LLM verify each claim as supported/contradicted. If no KB exists, returns a 0.5 penalty score.

### 3. Score Aggregation

```python
final_score = (
    w_rag   * rag_score   +    # 40% — highest weight, most reliable signal
    w_judge * judge_score  +    # 25% — fast, flexible, but judge can hallucinate
    w_se    * entropy_score +   # 20% — measures confidence, not correctness
    w_nli   * nli_score         # 15% — reserved for future NLI detector
) * 100
```

Weights are re-normalised based on which detectors actually ran.

### 4. Penalty Signals

Applied post-aggregation:
- **Overconfidence** (-5): No hedging language in responses > 200 chars
- **Length mismatch** (-3): Unusually long response to a simple query
- **URL presence**: Flagged for future HTTP validation

### 5. Threshold Gate

- **Score ≥ threshold** → Trustworthy. Delivered with green badge.
- **Score < threshold** → Flagged. Yellow/red banner with "Fix" button.

### 6. Correction Pipeline

Three strategies, chosen by the user or auto-selected:

| Strategy | Mechanism | Cost |
|---|---|---|
| Constrained re-prompting | Lists unverified claims and asks LLM to revise | 1 LLM call |
| RAG-grounded regeneration | Regenerates answer using only retrieved docs | 1 LLM call |
| Critic + Generator loop | Multi-round refinement (2–3 rounds) | 4–6 LLM calls |

## Key Design Decisions

### Model-Agnostic Wrapper
Every LLM call routes through `LLMClient`. To swap providers (Groq → OpenAI → Anthropic), change only `llm_client.py`. Detection logic never touches provider APIs directly.

### CPU-First Embedding
Using fastembed (ONNX runtime) instead of PyTorch-based sentence-transformers. This avoids 2GB+ RAM overhead and works on low-spec machines (i3, no GPU).

### Defensive JSON Parsing
LLMs often wrap JSON in markdown fences or add prose. `LLMClient._extract_json()` tries three parsing strategies before falling back.

### Graceful Degradation
If ChromaDB has no data, RAG returns 0.5 (not 0.0 or 1.0). Missing detectors are excluded from aggregation rather than blocking the pipeline.
