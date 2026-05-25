# ChestXRAI

AI-powered chest X-ray triage system. A DenseNet121 classifier detects 14 pathologies from the NIH ChestX-ray14 dataset, then a multi-agent CrewAI pipeline reviews the scan, validates GradCAM attention maps with a vision model, retrieves clinical guidelines, and generates a physician-ready report — all streamed live to a React dashboard.

---

## Deployment

### Frontend → Vercel

The React dashboard is deployed from this repo via Vercel. Every push to `main` triggers an automatic redeploy.

**To connect your own fork:**

1. Push this repo to GitHub.
2. Go to [vercel.com/new](https://vercel.com/new) → **Import Git Repository**.
3. Select the repo. Vercel auto-detects `vercel.json` — no extra config needed.
4. Add environment variables in the Vercel dashboard:

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | Your backend URL, e.g. `https://chestxrai-api.railway.app` |
| `VITE_WS_URL` | Same host with `wss://`, e.g. `wss://chestxrai-api.railway.app` |

5. Deploy. The frontend goes live; it connects to your self-hosted backend.

> **Why not all on Vercel?** The backend requires a persistent process (async queue worker), WebSocket server, PyTorch/DenseNet121 inference, and disk-based file storage. These don't fit the Vercel serverless model. The backend must run on a container platform (Railway, Render, Fly.io, or a plain VPS).

### Backend → Railway / Render / Fly.io (recommended)

**Railway** (simplest):

1. Create a new Railway project → **Deploy from GitHub repo**.
2. Set the start command to:
   ```
   cd api && uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
3. Add the same environment variables as your `.env` file (see [Configuration](#configuration)).
4. Copy the generated Railway domain → paste it as `VITE_API_URL` / `VITE_WS_URL` in Vercel.

**Render**:

1. New → **Web Service** → connect repo.
2. Root directory: `api`, start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. Add env vars, deploy.

### Local development

```bash
# Terminal 1 — backend
cd api
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev          # http://localhost:5173
```

No env vars needed locally — the frontend defaults to `http://localhost:8000`.

---

## Architecture

```
Upload X-ray
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI  (api/main.py)                                 │
│                                                         │
│  1. Radiologist Tool   — DenseNet121 inference          │
│                          GradCAM heatmaps saved         │
│                                                         │
│  2. Clinical Advisor   — Guideline lookup per finding   │
│                                                         │
│  3. CrewAI Hierarchical Pipeline                        │
│       Orchestrator (manager LLM)                        │
│       ├── Radiologist Agent  → X-ray classification     │
│       ├── VLM Reviewer Agent → Vision model inspection  │
│       ├── Clinical Advisor   → Protocol retrieval       │
│       └── Report Generator  → Narrative synthesis       │
└─────────────────────────────────────────────────────────┘
     │  WebSocket log stream
     ▼
React Dashboard (frontend/)
  ├── Patient queue (severity-sorted)
  ├── Triage report + confidence scores
  ├── GradCAM overlays
  ├── Agent reasoning log (collapsible per scan)
  └── Run logs page (full per-run history + AI narrative)
```

**Model**: DenseNet121 trained with the HERD optimizer on NIH ChestX-ray14 (112 000 images, 14 pathology classes). Pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.

---

## Prerequisites (backend)

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Node.js | 18+ |

For local models only:

| Requirement | Notes |
|-------------|-------|
| [Ollama](https://ollama.com) | latest |

```bash
ollama pull mistral:7b       # specialist agents
ollama pull qwen2.5:14b      # orchestrator
ollama pull llava:7b         # vision model (VLM review)
```

---

## Installation

### Backend

```bash
pip install crewai crewai-tools fastapi uvicorn python-multipart torch torchvision Pillow

# Optional — only needed for external providers you intend to use
pip install openai anthropic google-generativeai
```

### Frontend

```bash
cd frontend
npm install
```

---

## Configuration

All LLM settings are controlled by environment variables. Create a `.env` file at the project root or export them in your shell.

### Agent LLMs

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_LLM_PROVIDER` | `ollama` | Provider for specialist agents |
| `AGENT_LLM_MODEL` | `mistral:7b` | Model for specialist agents |
| `MANAGER_LLM_PROVIDER` | `ollama` | Provider for the orchestrator |
| `MANAGER_LLM_MODEL` | `qwen2.5:14b` | Model for the orchestrator |

### Vision Model (VLM)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_PROVIDER` | `ollama` | Provider for visual review |
| `VLM_MODEL` | `llava:7b` | Vision model |

### API Keys (for external providers)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GROQ_API_KEY` | Groq API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter key (gives access to all models) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |

### Supported Providers

| Provider | Agent example | VLM example |
|----------|--------------|-------------|
| `ollama` | `mistral:7b`, `qwen2.5:14b` | `llava:7b` |
| `openai` | `gpt-4o-mini`, `gpt-4o` | `gpt-4o` |
| `anthropic` | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` |
| `groq` | `llama-3.3-70b-versatile` | — |
| `gemini` | `gemini-2.5-flash`, `gemini-2.5-pro` | `gemini-2.5-flash` |
| `openrouter` | `google/gemini-2.5-flash` | `google/gemini-2.5-flash` |

**Quickstart with OpenRouter** (recommended for cloud deployment — one key for all models):

```bash
AGENT_LLM_PROVIDER=openrouter
AGENT_LLM_MODEL=google/gemini-2.5-flash
MANAGER_LLM_PROVIDER=openrouter
MANAGER_LLM_MODEL=google/gemini-2.5-flash
VLM_PROVIDER=openrouter
VLM_MODEL=google/gemini-2.5-flash
OPENROUTER_API_KEY=sk-or-v1-...
```

**Gemini directly:**

```bash
AGENT_LLM_PROVIDER=gemini
AGENT_LLM_MODEL=gemini-2.5-flash
MANAGER_LLM_PROVIDER=gemini
MANAGER_LLM_MODEL=gemini-2.5-flash
VLM_PROVIDER=gemini
VLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=AIza...
```

---

## How it works

1. **Upload** a chest X-ray via the dashboard.
2. **DenseNet121** runs multi-label inference and generates GradCAM heatmaps for each detected pathology (saved to `src/chestxrai/gradcam_outputs/`).
3. **Clinical Advisor** queries a local guideline database for each detected pathology and returns urgency level and follow-up protocols.
4. **CrewAI hierarchical pipeline** kicks off:
   - The **orchestrator** delegates work to four specialist agents.
   - **Radiologist agent** calls the triage tool for a structured classification summary.
   - **VLM Reviewer agent** sends the original scan + top GradCAM overlays to a vision model, which validates whether the highlighted regions align with the predicted pathology.
   - **Clinical Advisor agent** retrieves evidence-based protocols.
   - **Report Generator agent** synthesises all findings into a physician-ready narrative.
5. All agent reasoning is streamed in real time to the **Agent Reasoning** panel via WebSocket.
6. The **Run Logs** page (click "Logs ↗" in the Agent Reasoning panel) shows the full execution history per scan, including the AI narrative report from the crew.
7. The final report lands in the **Physician Review** queue, where it can be approved or rejected.

---

## Project structure

```
chestxrai/
├── vercel.json              # Vercel build config (frontend only)
├── .env                     # Local secrets — never committed
│
├── api/
│   ├── main.py              # FastAPI app, queue worker, WebSocket log stream
│   └── uploads/             # Uploaded X-ray files (gitignored)
│
├── src/chestxrai/
│   ├── crew.py              # CrewAI crew definition (hierarchical pipeline)
│   ├── llm_config.py        # Provider-agnostic LLM factory (env-var driven)
│   ├── config/
│   │   ├── agents.yaml      # Agent roles and goals
│   │   └── tasks.yaml       # Task descriptions and context chains
│   ├── tools/
│   │   ├── triage_tool.py   # DenseNet121 inference + GradCAM
│   │   ├── guideline_tool.py# Clinical guideline lookup
│   │   └── vlm_tool.py      # Vision model review (Ollama / OpenAI / Gemini / OpenRouter)
│   └── gradcam_outputs/     # Generated GradCAM heatmap PNGs (gitignored)
│
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main dashboard layout
    │   ├── constants.js      # API_BASE / WS_BASE (reads VITE_API_URL / VITE_WS_URL)
    │   ├── components/
    │   │   ├── AgentLog.jsx  # Live agent reasoning panel
    │   │   ├── RunLogs.jsx   # Full per-run log page
    │   │   ├── TriageReport.jsx # Paper-style radiology report + AI narrative
    │   │   ├── XRayViewer.jsx
    │   │   ├── PatientQueue.jsx
    │   │   └── TopBar.jsx
    │   └── hooks/
    │       ├── useWebSocket.js
    │       └── usePatients.js
    └── package.json
```

---

## Notes

- The DenseNet121 model weights are loaded once at first inference and cached for the process lifetime.
- GradCAM overlays are generated for all pathologies with confidence ≥ 15%; the VLM tool picks the top 2.
- For GPU-accelerated inference, ensure PyTorch is installed with CUDA/MPS support.
- The system uses ~13 GB RAM when running Mistral 7B + Qwen2.5 14B via Ollama. For cloud deployment, use OpenRouter or Gemini to avoid local model requirements.
