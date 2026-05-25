# ChestXRAI

AI-powered chest X-ray triage system. A DenseNet121 classifier detects 14 pathologies from the NIH ChestX-ray14 dataset, then a multi-agent CrewAI pipeline reviews the scan, validates GradCAM attention maps with a vision model, retrieves clinical guidelines, and generates a physician-ready report — all streamed live to a React dashboard.

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
│       Orchestrator (qwen2.5:14b)                        │
│       ├── Radiologist Agent  → X-ray classification     │
│       ├── VLM Reviewer Agent → LLaVA visual inspection  │
│       ├── Clinical Advisor   → Protocol retrieval        │
│       └── Report Generator  → Narrative synthesis       │
└─────────────────────────────────────────────────────────┘
     │  WebSocket log stream
     ▼
React Dashboard (frontend/)
  ├── Patient queue (severity-sorted)
  ├── Triage report + confidence scores
  ├── GradCAM overlays
  └── Agent reasoning log (collapsible per scan)
```

**Model**: DenseNet121 trained with the HERD optimizer on NIH ChestX-ray14 (112 000 images, 14 pathology classes). Pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Node.js | 18+ |
| [Ollama](https://ollama.com) | latest |

Pull the required models:

```bash
ollama pull mistral:7b       # specialist agents
ollama pull qwen2.5:14b      # orchestrator
ollama pull llava:7b         # vision model (VLM review)
```

---

## Installation

### Backend

```bash
cd chestxrai
pip install crewai crewai-tools fastapi uvicorn python-multipart torch torchvision Pillow
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
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |

### Supported Providers

| Provider | Agent example | VLM example |
|----------|--------------|-------------|
| `ollama` | `mistral:7b`, `qwen2.5:14b` | `llava:7b` |
| `openai` | `gpt-4o-mini`, `gpt-4o` | `gpt-4o` |
| `anthropic` | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` |
| `groq` | `llama-3.3-70b-versatile` | — |

Example — use OpenAI for agents and Anthropic for VLM:

```bash
export AGENT_LLM_PROVIDER=openai
export AGENT_LLM_MODEL=gpt-4o-mini
export MANAGER_LLM_PROVIDER=openai
export MANAGER_LLM_MODEL=gpt-4o
export VLM_PROVIDER=anthropic
export VLM_MODEL=claude-haiku-4-5-20251001
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

### 1. Start Ollama (if using local models)

```bash
ollama serve
```

### 2. Start the backend

```bash
cd api
uvicorn main:app --reload --port 8000
```

### 3. Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

---

## How it works

1. **Upload** a chest X-ray via the dashboard.
2. **DenseNet121** runs multi-label inference and generates GradCAM heatmaps for each detected pathology (saved to `src/chestxrai/gradcam_outputs/`).
3. **Clinical Advisor** queries a local guideline database for each detected pathology and returns urgency level and follow-up protocols.
4. **CrewAI hierarchical pipeline** kicks off:
   - The **orchestrator** (`qwen2.5:14b` by default) delegates work to four specialist agents.
   - **Radiologist agent** calls the triage tool for a structured classification summary.
   - **VLM Reviewer agent** sends the original scan + top GradCAM overlays to a vision model, which validates whether the highlighted regions align with the predicted pathology.
   - **Clinical Advisor agent** retrieves evidence-based protocols.
   - **Report Generator agent** synthesises all findings into a physician-ready narrative.
5. All agent reasoning is streamed in real time to the **Agent Reasoning** panel via WebSocket.
6. The final report lands in the **Physician Review** queue, where it can be approved or rejected.

---

## Project structure

```
chestxrai/
├── api/
│   ├── main.py              # FastAPI app, queue worker, WebSocket log stream
│   └── uploads/             # Uploaded X-ray files
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
│   │   └── vlm_tool.py      # Vision model review (Ollama / OpenAI / Anthropic)
│   └── gradcam_outputs/     # Generated GradCAM heatmap PNGs
│
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main dashboard layout
    │   ├── components/
    │   │   └── AgentLog.jsx  # Live agent reasoning panel
    │   └── hooks/
    │       └── useWebSocket.js
    └── package.json
```

---

## Notes

- The DenseNet121 model weights are loaded once at first inference and cached for the process lifetime.
- GradCAM overlays are generated for all pathologies with confidence ≥ 15%; the VLM tool picks the top 2.
- The system requires **~13 GB RAM** to run Mistral 7B + Qwen2.5 14B simultaneously via Ollama (tested on Apple M4 16 GB).
- For GPU-accelerated inference, ensure PyTorch is installed with CUDA/MPS support.
