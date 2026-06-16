# Hallucination Detector

A multi-signal middleware that evaluates LLM response trustworthiness and optionally fixes hallucinated content.

## What It Does

Sits between your user and any LLM. You send a query + response — it scores trustworthiness (0–100) using four independent detection signals, and offers correction when scores are low.

### Detection Signals

| Detector | What It Checks | Weight |
|---|---|---|
| **RAG Grounding** | Claim verification against live Google Search results (Serper.dev) | 40% |
| **LLM Judge** | Structured fact-checking via a stronger judge model | 25% |
| **Semantic Entropy** | Self-consistency across multiple samples using cosine similarity | 20% |
| **NLI Entailment** | Natural language inference-based entailment scoring | 15% |

### Score Interpretation

- 🟢 **75–100** — Trustworthy. Response is consistent and grounded.
- 🟡 **50–74** — Uncertain. Some claims may be unverifiable.
- 🔴 **0–49** — Likely Hallucinated. Multiple signals disagree or flag issues.

## Score Calibration & Logic

To prevent deceptive metrics (like high confidence for incorrect facts), the system uses a calibrated blending approach:

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

### 🔴 RAG Grounding (Live Web Search)
Claims are extracted from the response and verified against real-time Google Search results:
- Each claim is searched independently and concurrently via the Serper.dev API.
- Verdicts: `supported`, `partially_supported`, `contradicted`, or `not_found`.
- Scoring: `supported=1.0`, `partial=0.5`, `not_found=0.3`, `contradicted=0.0`.
- If `SERPER_API_KEY` is not set, returns a neutral `0.5` penalty score.

## Tech Stack

- **Backend**: FastAPI + Python
- **Frontend**: React + Vite + Tailwind CSS v4
- **LLM (Generation)**: Groq — `llama-3.1-8b-instant` (fast, for generating responses)
- **LLM (Judge)**: Groq — `llama-3.3-70b-versatile` (stronger, for fact-checking)
- **Web Search**: Serper.dev (real-time Google Search for RAG grounding)
- **HTTP Client**: httpx (async web search requests)

## Screenshots

<img width="1918" height="992" alt="image" src="https://github.com/user-attachments/assets/c5ec566d-f056-482b-b413-b874524742d9" />
<img width="1918" height="987" alt="image" src="https://github.com/user-attachments/assets/031f3e84-592e-49a9-80b6-e36d2c3f54f6" />
<img width="1080" height="718" alt="image" src="https://github.com/user-attachments/assets/d90dafd6-d993-45d9-99df-f18608b5e6b3" />
<img width="1335" height="717" alt="image" src="https://github.com/user-attachments/assets/6f53444f-ac57-4028-987d-8ac6bd4148ed" />
<img width="1297" height="732" alt="image" src="https://github.com/user-attachments/assets/f5b28bbd-c483-4508-b9f0-3a9cabf7b113" />
<img width="1488" height="798" alt="image" src="https://github.com/user-attachments/assets/24b7483c-43f2-4063-bf5d-585d36e344ec" />
<img width="1475" height="790" alt="image" src="https://github.com/user-attachments/assets/3ca5cf78-7062-49a7-9764-6de80386abab" />

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
# Edit .env and add your API keys (see Environment Variables below)
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
| `GET` | `/api/health` | Health check + model info (shows both generation and judge models) |
| `POST` | `/api/detect` | Score a query+response pair |
| `POST` | `/api/chat` | Generate → detect → optionally fix |
| `POST` | `/api/chat/stream` | Streaming version of `/api/chat` |
| `POST` | `/api/correct` | Fix a hallucinated response |


## Correction Strategies

| Strategy | Description | When to Use |
|---|---|---|
| `constrained` | Re-prompts the LLM to revise specific unverified claims | Default first pass |
| `rag` | Regenerates answer grounded exclusively in retrieved web snippets | When claims are web-verifiable |
| `critic_loop` | Multi-round critic + generator refinement (2–3 rounds) | High-stakes questions |

## Project Structure

```
hallucination-detector/
├── backend/
│   ├── main.py                  # FastAPI app + endpoints
│   ├── llm_client.py            # Model-agnostic LLM wrapper (Groq default)
│   ├── aggregator.py            # Weighted scoring + penalties
│   ├── corrector.py             # Fix pipeline (3 strategies)
│   ├── cache_manager.py         # Request-level caching
│   └── detectors/
│       ├── base.py              # Abstract base class + DetectorResult
│       ├── semantic_entropy.py  # Self-consistency via cosine similarity
│       ├── llm_judge.py         # LLM-as-judge fact-checking (stronger model)
│       ├── rag_grounding.py     # Live web search claim verification (Serper.dev)
│       └── nli_entailment.py    # NLI-based entailment scoring
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
| `GROQ_API_KEY` | — | Your Groq API key (also supports `GROQ_API_KEYS` as a comma-separated list for key rotation) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model used to generate responses |
| `JUDGE_MODEL` | `llama-3.3-70b-versatile` | Stronger model used exclusively for LLM-as-judge fact-checking |
| `DETECTION_THRESHOLD` | `65` | Score threshold below which a response is flagged |
| `SELF_CONSISTENCY_SAMPLES` | `3` | Number of samples generated for semantic entropy check |
| `SERPER_API_KEY` | — | Serper.dev API key for live Google Search grounding (free tier available at serper.dev) |

## License

MIT
