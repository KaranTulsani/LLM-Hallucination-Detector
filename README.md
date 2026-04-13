# Hallucination Detector

A multi-signal middleware that evaluates LLM response trustworthiness and optionally fixes hallucinated content.

## What It Does

Sits between your user and any LLM. You send a query + response — it scores trustworthiness (0–100) using three independent detection signals, and offers correction when scores are low.

### Detection Signals

| Detector | What It Checks | Weight |
|---|---|---|
| **Semantic Entropy** | Self-consistency across 5 samples using cosine similarity | 20% |
| **LLM Judge** | Structured fact-checking via a judge prompt | 25% |
| **RAG Grounding** | Claim verification against a knowledge base | 40% |

### Score Interpretation

- 🟢 **75–100** — Trustworthy. Response is consistent and grounded.
- 🟡 **50–74** — Uncertain. Some claims may be unverifiable.
- 🔴 **0–49** — Likely Hallucinated. Multiple signals disagree or flag issues.

## Score Calibration & Logic

To prevent deceptive metrics (like high high confidence for incorrect facts), the system uses a calibrated blending approach:

### 🟢 Semantic Entropy (Threshold-based)
Instead of a raw similarity score, we use a calibrated threshold mapping:
- **Similarity > 0.92:** Mapped to **100%** (Full Confidence).
- **Similarity < 0.60:** Mapped to **0%** (No Confidence).
- **Why?** Slight phrasing variations in highly consistent answers (e.g., using synonyms) naturally prevent a perfect 1.0 cosine similarity. A value like **0.98** is already considered "perfectly consistent."

### 🟡 LLM Judge (Weighted Blend + Penalties)
The Judge score is a holistic evaluation, not just a raw confidence number:
- **60% Weight:** Claim Verification Ratio (Verified vs Unverified claims).
- **40% Weight:** Judge's internal confidence score.
- **Penalties:** Subtracts points for "Overconfidence flags" (e.g., certain language for uncertain topics) or "Fabricated Citations."

*Example: A judge with 90% confidence that verified 4/5 claims but used overconfident language results in a final score of **79%**.*

## Tech Stack

- **Backend**: FastAPI + Python
- **Frontend**: React + Vite + Tailwind CSS v4
- **LLM**: Groq (model-agnostic wrapper — swap to any provider)
- **Vector DB**: ChromaDB (persistent, ONNX embeddings via fastembed)
- **Embeddings**: fastembed (BAAI/bge-small-en-v1.5, CPU-only, no PyTorch)

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
python -m uvicorn backend.main:app --reload
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/api` to the backend at `http://localhost:8000`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + model info |
| `POST` | `/api/detect` | Score a query+response pair |
| `POST` | `/api/chat` | Generate → detect → optionally fix |
| `POST` | `/api/correct` | Fix a hallucinated response |
| `POST` | `/api/kb/add` | Add documents to knowledge base |
| `DELETE` | `/api/kb/reset` | Clear the knowledge base |

### Example: Detect Hallucination

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When was the Eiffel Tower built?",
    "response": "The Eiffel Tower was built in 1901."
  }'
```

### Example: Full Chat Flow

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who invented the telephone?",
    "auto_fix": false
  }'
```

## Correction Strategies

| Strategy | Description | When to Use |
|---|---|---|
| `constrained` | Re-prompts the LLM to revise specific unverified claims | Default first pass |
| `rag` | Regenerates answer grounded exclusively in retrieved docs | When knowledge base is populated |
| `critic_loop` | Multi-round critic + generator refinement (2–3 rounds) | High-stakes questions |

## Project Structure

```
hallucination-detector/
├── backend/
│   ├── main.py                  # FastAPI app + endpoints
│   ├── llm_client.py            # Model-agnostic LLM wrapper (Groq default)
│   ├── aggregator.py            # Weighted scoring + penalties
│   ├── corrector.py             # Fix pipeline (3 strategies)
│   └── detectors/
│       ├── base.py              # Abstract base class + DetectorResult
│       ├── semantic_entropy.py  # Self-consistency via embeddings
│       ├── llm_judge.py         # LLM-as-judge fact-checking
│       └── rag_grounding.py     # RAG-based claim verification
└── frontend/
    ├── src/
    │   ├── App.jsx              # Main layout + state management
    │   ├── api.js               # API client
    │   ├── index.css            # Design system
    │   └── components/
    │       ├── ChatInput.jsx    # Auto-resizing input
    │       └── ResponseCard.jsx # Score badge + evidence panel
    └── vite.config.js           # Vite + Tailwind + API proxy
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model to use |
| `DETECTION_THRESHOLD` | `65` | Score threshold for trustworthiness |
| `SELF_CONSISTENCY_SAMPLES` | `5` | Number of samples for semantic entropy |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage path |

## License

MIT
