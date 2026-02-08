"""
Financial Audio Intelligence API
Full pipeline: Audio → Text Processing → Backboard Intelligence
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
import shutil
import json
from groq import Groq

from layer1_audio import run_layer1
from layer2_text import run_layer2, load_policy_rules
from layer3_backboard import initialize_assistants, run_layer3
from pipeline import run_full_pipeline, save_report

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Financial Audio Intelligence API",
    version="2.0.0",
    description="4-Layer compliance analysis for financial service call recordings",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders
UPLOAD_DIR = Path("../data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm"}

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Startup: initialize Backboard assistants ──────────────────────────────

@app.on_event("startup")
async def startup():
    """Create Backboard assistants and upload policy docs on server start."""
    try:
        await initialize_assistants()
        print("✓ Backboard assistants initialized")
    except Exception as e:
        print(f"⚠ Backboard init deferred: {e}")


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Financial Audio Intelligence API v2",
        "layers": [
            "Layer 1: Audio Forensics (Groq Whisper + Librosa)",
            "Layer 2: Text Processing (spaCy + PII + Profanity)",
            "Layer 3: Intelligence (Backboard.io — 3 Assistants)",
            "Layer 4: Review UI (Next.js)",
        ],
        "timestamp": datetime.now().isoformat(),
    }


# ── Upload only ───────────────────────────────────────────────────────────

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file without processing."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    dst = UPLOAD_DIR / safe_name

    with open(dst, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    return {
        "status": "uploaded",
        "filename": safe_name,
        "size_mb": round(dst.stat().st_size / (1024 * 1024), 2),
        "path": str(dst),
    }


# ── Layer 1 only: Transcribe + Quality ─────────────────────────────────────

@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Layer 1 only: Transcribe audio + audio quality analysis."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    dst = UPLOAD_DIR / safe_name

    try:
        with open(dst, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        result = run_layer1(str(dst), groq_client)
        result["filename"] = safe_name
        result["status"] = "success"
        return result

    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")


# ── Layer 2 only: Text analysis on raw text ────────────────────────────────

@app.post("/analyze-text")
async def analyze_text(body: dict):
    """Layer 2 only: Run text processing on provided transcript text."""
    transcript = body.get("transcript", "")
    if not transcript.strip():
        raise HTTPException(400, "No transcript provided")

    result = run_layer2(transcript)
    result["status"] = "success"
    return result


# ── Full Pipeline: Audio → L1 → L2 → L3 ──────────────────────────────────

@app.post("/analyze")
async def full_analysis(file: UploadFile = File(...), language: str | None = None):
    """
    FULL PIPELINE: Upload audio → Layer 1 → Layer 2 → Layer 3.
    Returns complete compliance report.
    Optionally specify language (e.g. 'en', 'ru', 'hi') to improve accuracy.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    dst = UPLOAD_DIR / safe_name

    try:
        with open(dst, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        report = await run_full_pipeline(str(dst), groq_client, language=language)

        # Save report
        report_path = await save_report(report)
        report["report_path"] = report_path

        return report

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


# ── Get policy rules ──────────────────────────────────────────────────────

@app.get("/policies")
def get_policies():
    """Return loaded policy documents."""
    rules = load_policy_rules()
    return {"policies": rules}


# ── List saved reports ─────────────────────────────────────────────────────

@app.get("/reports")
def list_reports():
    """List all saved analysis reports."""
    output_dir = Path("../data/outputs")
    if not output_dir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(output_dir.glob("report_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            reports.append({
                "filename": f.name,
                "audio_file": data.get("audio_file", ""),
                "processed_at": data.get("processed_at", ""),
                "overall_risk": data.get("overall_risk", {}),
                "duration": data.get("duration_seconds"),
            })
        except Exception:
            continue

    return {"reports": reports}


@app.get("/reports/{filename}")
def get_report(filename: str):
    """Get a specific saved report."""
    filepath = Path("../data/outputs") / filename
    if not filepath.exists():
        raise HTTPException(404, "Report not found")

    return json.loads(filepath.read_text())


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
