import uuid
import sys
import asyncio
import queue as _std_queue
import json
import re
import io
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from collections import OrderedDict
import os
from dotenv import load_dotenv
from PIL import Image as PILImage

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Resolve paths relative to this file so uvicorn can be launched from any cwd
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_CHESTXRAI_DIR = os.path.join(_API_DIR, "..", "src", "chestxrai")

# JSON log file
_LOG_DIR = os.path.join(_API_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_JSON_LOG_PATH = os.path.join(_LOG_DIR, "agent_log.jsonl")

def _write_json_log(message: str) -> None:
    m = re.match(r"^\[([^\]]+)\]\s*(.*)", message, re.DOTALL)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": m.group(1) if m else "System",
        "message": m.group(2).strip() if m else message,
    }
    with open(_JSON_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# Make the tools package importable
sys.path.insert(0, os.path.abspath(_CHESTXRAI_DIR))

patients: OrderedDict = OrderedDict()
log_connections: list[WebSocket] = []
queue: asyncio.Queue = asyncio.Queue()


async def broadcast_log(message: str):
    _write_json_log(message)
    dead = []
    for ws in log_connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        log_connections.remove(ws)


async def process_queue():
    await broadcast_log("[Queue] Worker online — listening for incoming scans...")

    while True:
        patient_id = await queue.get()
        patient = patients[patient_id]
        patient["status"] = "analyzing"
        patient.setdefault("run_log", [])

        # Helper: broadcast to WebSocket AND record in this patient's run_log
        async def record(msg: str):
            patient["run_log"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": msg,
            })
            await broadcast_log(msg)

        await record(f"[Queue] ▶ Dispatching patient {patient_id}: {patient['filename']}")
        await record(f"[Queue] Pipeline: Radiologist → VLM Review → Clinical Advisor → Report")

        try:
            from tools.triage_tool import analyze_xray
            from tools.guideline_tool import lookup_guideline
            from crew import ChestXRAICrew

            # ── Radiologist — structured inference ────────────────
            await record(f"[Radiologist] Received scan: {patient['filename']}")
            await record(f"[Radiologist] Loading DenseNet121 (NIH ChestX-ray14, 112K images)...")
            await record(f"[Radiologist] Running multi-label inference across 14 pathology classes...")

            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, analyze_xray, patient["filepath"])

            detected   = result.get("detected_pathologies", [])
            borderline = result.get("borderline_pathologies", [])
            all_preds  = result.get("all_predictions", {})

            for name, prob in sorted(all_preds.items(), key=lambda x: -x[1]):
                if prob > 0.30:
                    await record(f"[Radiologist] ✦ {name}: {prob:.1%} — DETECTED (≥30%)")
                elif prob > 0.15:
                    await record(f"[Radiologist] ◦ {name}: {prob:.1%} — borderline (15–30%)")

            if not detected and not borderline:
                await record(f"[Radiologist] No pathologies above detection threshold — scan appears clear")

            await record(
                f"[Radiologist] Assessment: severity={result['severity']} | "
                f"primary={result['top_finding']} ({result['top_finding_confidence']:.1%})"
            )

            patient["analysis"] = result

            # ── Clinical Advisor — guideline lookup ───────────────
            if detected:
                await record(
                    f"[Clinical Advisor] {len(detected)} patholog{'y' if len(detected)==1 else 'ies'} "
                    f"flagged — querying clinical guideline database..."
                )
            else:
                await record(f"[Clinical Advisor] No detected pathologies — skipping guideline lookup")

            guidelines = []
            for p in detected:
                await record(f"[Clinical Advisor] Querying: {p['pathology']}...")
                g = lookup_guideline(p["pathology"])
                guidelines.append(g)
                if g.get("found"):
                    await record(
                        f"[Clinical Advisor] {p['pathology']}: urgency={g.get('urgency','?').upper()} | "
                        f"{len(g.get('follow_up', []))} follow-up protocol(s)"
                    )

            patient["guidelines"] = guidelines

            # ── Crew — orchestrated pipeline (hierarchical) ────────
            import crew as _crew_module

            _manager = f"{os.getenv('MANAGER_LLM_PROVIDER','ollama')}:{os.getenv('MANAGER_LLM_MODEL','qwen2.5:14b')}"
            await record(f"[Orchestrator] {_manager} online — starting hierarchical pipeline...")

            crew_log_q = _std_queue.SimpleQueue()
            _crew_module._active_log_q = crew_log_q

            # Drain crew step-callback messages → WebSocket + run_log
            async def _drain():
                while True:
                    while True:
                        try:
                            await record(crew_log_q.get_nowait())
                        except _std_queue.Empty:
                            break
                    await asyncio.sleep(0.1)

            drain_task = asyncio.create_task(_drain())
            try:
                crew_obj    = ChestXRAICrew().crew()
                crew_result = await loop.run_in_executor(
                    None,
                    lambda: crew_obj.kickoff(inputs={"image_path": patient["filepath"]}),
                )
                patient["report"] = crew_result.raw
            finally:
                _crew_module._active_log_q = None
                await asyncio.sleep(0.3)
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass
                while True:
                    try:
                        await record(crew_log_q.get_nowait())
                    except _std_queue.Empty:
                        break

            severity_order = {"CRITICAL": 0, "ABNORMAL": 1, "NORMAL": 2}
            patient["sort_key"] = severity_order.get(result["severity"], 2)
            patient["status"]   = "awaiting_review"

            await record(f"[Report] ✓ Report ready — dispatched to physician review queue")

        except Exception as e:
            patient["status"] = "error"
            patient["error"]  = str(e)
            await record(f"[Error] Processing failed for {patient['filename']}: {e}")

        queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(process_queue())
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded X-rays and GradCAM heatmaps as static files
_uploads_dir = os.path.join(_API_DIR, "uploads")
_gradcam_dir  = os.path.abspath(os.path.join(_CHESTXRAI_DIR, "gradcam_outputs"))
os.makedirs(_uploads_dir, exist_ok=True)
os.makedirs(_gradcam_dir,  exist_ok=True)
app.mount("/uploads", StaticFiles(directory=_uploads_dir), name="uploads")
app.mount("/gradcam",  StaticFiles(directory=_gradcam_dir),  name="gradcam")


@app.post("/upload")
async def upload_xray(file: UploadFile):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        return {"error": "Unsupported format. Upload PNG or JPEG only."}

    content = await file.read()
    try:
        PILImage.open(io.BytesIO(content)).verify()
    except Exception:
        return {"error": "File is not a valid image."}

    patient_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(_uploads_dir, f"{patient_id}_{file.filename}")

    with open(filepath, "wb") as f:
        f.write(content)

    patients[patient_id] = {
        "id": patient_id,
        "filename": file.filename,
        "filepath": filepath,
        "status": "queued",
        "timestamp": datetime.now().isoformat(),
        "analysis": None,
        "guidelines": None,
        "report": None,
        "sort_key": 2,
        "run_log": [],
    }

    await broadcast_log(f"[Queue] New patient {patient_id}: {file.filename}")

    await queue.put(patient_id)
    return {"patient_id": patient_id, "status": "queued"}


@app.get("/queue")
async def get_queue():
    # Exclude run_log from queue listing to keep the response lightweight
    sorted_patients = sorted(
        patients.values(), key=lambda p: (p["sort_key"], p["timestamp"])
    )
    return [
        {k: v for k, v in p.items() if k != "run_log"}
        for p in sorted_patients
    ]


@app.get("/patient/{patient_id}")
async def get_patient(patient_id: str):
    if patient_id not in patients:
        return {"error": "not found"}
    return patients[patient_id]


@app.post("/review/{patient_id}")
async def review_patient(patient_id: str, decision: str, notes: str = ""):
    if patient_id not in patients:
        return {"error": "not found"}

    patient = patients[patient_id]
    patient["review"] = {
        "decision": decision,
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }
    patient["status"] = "approved" if decision == "approve" else "rejected"
    await broadcast_log(f"[HITL] {patient['filename']} → {decision}")

    return {"status": patient["status"]}


@app.websocket("/ws/logs")
async def websocket_logs(ws: WebSocket):
    await ws.accept()
    log_connections.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        if ws in log_connections:
            log_connections.remove(ws)
