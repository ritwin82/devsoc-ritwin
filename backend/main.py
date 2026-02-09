"""
Financial Audio Intelligence API
Full pipeline: Audio → Text Processing → Backboard Intelligence
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import os
import shutil
import json

from layer1_audio import run_layer1
from layer2_text import run_layer2, load_policy_rules
from layer3_backboard import initialize_assistants, run_layer3
from finbert_extractor import run_finbert_analysis, prepare_terms_for_explanation
from pipeline import run_full_pipeline, save_report, _compute_overall_risk
from realtime import handle_realtime_websocket
from memory_layer import get_memory_manager, generate_session_id

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Financial Audio Intelligence API",
    version="2.1.0",
    description="4-Layer compliance analysis for financial service call recordings with FinBERT-powered term extraction",
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
            "Layer 1: Audio Forensics (ElevenLabs Scribe + Librosa)",
            "Layer 2: Text Processing (spaCy + PII + Profanity)",
            "Layer 3: Intelligence (Backboard.io — 4 Assistants + FinBERT)",
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

        result = run_layer1(str(dst))
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


# ── FinBERT Term Analysis ────────────────────────────────────────────────────

@app.post("/analyze-terms")
async def analyze_financial_terms(body: dict):
    """
    Standalone FinBERT analysis: Extract financial terms and get LLM explanations.
    
    Input: {"transcript": "your transcript text here"}
    
    Returns:
    - finbert_analysis: Important segments and extracted terms
    - term_explanations: LLM-generated explanations for each term
    """
    transcript = body.get("transcript", "")
    if not transcript.strip():
        raise HTTPException(400, "No transcript provided")

    try:
        # Run FinBERT analysis
        finbert_result = run_finbert_analysis(transcript)
        financial_terms = prepare_terms_for_explanation(finbert_result)
        
        # Get LLM explanations if we have terms
        if financial_terms:
            from layer3_backboard import explain_financial_terms, initialize_assistants, _try_parse_json
            await initialize_assistants()
            term_explanations_raw = await explain_financial_terms(financial_terms)
            term_explanations = _try_parse_json(term_explanations_raw)
        else:
            term_explanations = {"term_explanations": [], "summary": "No financial terms found."}
        
        return {
            "status": "success",
            "finbert_analysis": finbert_result,
            "term_explanations": term_explanations,
        }
    except Exception as e:
        raise HTTPException(500, f"Term analysis failed: {e}")


# ── SSE helper ─────────────────────────────────────────────────────────────

def _sse(event_type: str, stage: str, message: str) -> str:
    """Format a Server-Sent Event line."""
    return f"data: {json.dumps({'type': event_type, 'stage': stage, 'message': message})}\n\n"


# ── Full Pipeline: Audio → L1 → L2 → L3 (SSE streaming) ──────────────────

@app.post("/analyze")
async def full_analysis(
    file: UploadFile = File(...), 
    language: str | None = None,
    caller_id: str | None = None,
    session_id: str | None = None
):
    """
    FULL PIPELINE: Upload audio → Layer 1 → Layer 2 → Layer 3.
    Returns complete compliance report with memory context.
    
    Args:
        file: Audio file to analyze
        language: Optional language hint (e.g. 'en', 'ru', 'hi')
        caller_id: Optional caller identifier for memory persistence across calls
        session_id: Optional session ID for grouping related analyses
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp_str}_{file.filename}"
    dst = UPLOAD_DIR / safe_name

    try:
        with open(dst, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        report = await run_full_pipeline(
            str(dst), groq_client, language,
            caller_id=caller_id,
            session_id=session_id
        )

        # Save report
        report_path = await save_report(report)
        report["report_path"] = report_path

        return report

    async def event_stream():
        try:
            start_time = datetime.now()

            # ── Layer 1: Transcription + Audio Forensics ──────────────
            yield _sse("progress", "layer1", "Layer 1: Transcribing audio with ElevenLabs Scribe & running audio forensics...")
            layer1 = await asyncio.to_thread(run_layer1, str(dst), language)
            transcript = layer1["transcript"]

            if not transcript or not transcript.strip() or len(transcript.strip()) < 10:
                yield _sse("error", "", "Transcription returned too short or empty — audio may be silent, corrupted, or in an unrecognized language. Try selecting the correct language.")
                return

            if not layer1.get("segments"):
                yield _sse("error", "", "No usable speech segments found — audio may be too noisy or the language may be incorrect. Try selecting the correct language.")
                return

            seg_count = len(layer1.get("segments", []))
            dur = round(layer1.get("duration", 0))
            yield _sse("progress", "layer1_done", f"Layer 1 complete — {seg_count} segments, {dur}s audio transcribed")

            # ── Layer 2: Text Analysis ────────────────────────────────
            yield _sse("progress", "layer2", "Layer 2: Scanning for PII, profanity, obligations & financial entities...")
            detected_lang = layer1.get("language", "en")
            layer2 = await asyncio.to_thread(run_layer2, transcript, detected_lang)
            pii_n = layer2.get("pii_count", 0)
            prof_n = len(layer2.get("profanity_findings", []))
            yield _sse("progress", "layer2_done", f"Layer 2 complete — {pii_n} PII items, {prof_n} profanity findings")

            # ── Layer 3: AI Compliance ────────────────────────────────
            yield _sse("progress", "layer3", "Layer 3: Running AI compliance analysis (Backboard.io + FinBERT)...")
            layer3 = await run_layer3(transcript, layer2)
            yield _sse("progress", "layer3_done", "Layer 3 complete — obligations, intent, compliance & terms analyzed")

            # ── Assemble Report ───────────────────────────────────────
            yield _sse("progress", "saving", "Computing risk score & saving report...")
            elapsed = (datetime.now() - start_time).total_seconds()

            report = {
                "status": "success",
                "processed_at": start_time.isoformat(),
                "processing_time_seconds": round(elapsed, 2),
                "audio_file": Path(str(dst)).name,
                # Layer 1
                "transcript": transcript,
                "language": detected_lang,
                "duration_seconds": layer1["duration"],
                "segments": layer1["segments"],
                "audio_quality": layer1["audio_quality"],
                "overall_confidence": layer1.get("overall_confidence"),
                "emotion_analysis": layer1.get("emotion_analysis"),
                "tamper_detection": layer1.get("tamper_detection"),
                # Layer 2
                "pii_detected": layer2["pii_detected"],
                "pii_count": layer2["pii_count"],
                "financial_entities": layer2["financial_entities"],
                "named_entities": layer2["named_entities"],
                "profanity_findings": layer2["profanity_findings"],
                "obligation_sentences": layer2["obligation_sentences"],
                "text_risk_level": layer2["risk_level"],
                # Layer 3
                "finbert_analysis": layer3.get("finbert_analysis"),
                "financial_term_explanations": layer3.get("term_explanations"),
                "obligation_analysis": layer3["obligation_analysis"],
                "intent_classification": layer3["intent_classification"],
                "regulatory_compliance": layer3["regulatory_compliance"],
                # Overall
                "overall_risk": _compute_overall_risk(layer1, layer2, layer3),
            }

            report_path = await save_report(report)
            report["report_path"] = report_path

            yield f"data: {json.dumps({'type': 'complete', 'report': report}, default=str)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Get policy rules ──────────────────────────────────────────────────────

@app.get("/policies")
def get_policies():
    """Return loaded policy documents."""
    rules = load_policy_rules()
    return {"policies": rules}


# ── Memory Management Endpoints ───────────────────────────────────────────

@app.get("/memory/caller/{caller_id}")
async def get_caller_memory(caller_id: str):
    """
    Get memory/history for a specific caller.
    Returns call history and patterns associated with the caller.
    """
    memory_manager = await get_memory_manager()
    history = await memory_manager.get_caller_history(caller_id)
    context = await memory_manager.get_caller_context(caller_id)
    
    return {
        "caller_id": caller_id,
        "call_count": len(history),
        "call_history": history,
        "context_summary": context,
    }


@app.get("/memory/session/{session_id}")
async def get_session_memory(session_id: str):
    """
    Get current session context.
    Returns context data accumulated during an analysis session.
    """
    memory_manager = await get_memory_manager()
    context = await memory_manager.get_session_context(session_id)
    
    return {
        "session_id": session_id,
        "context": context,
        "exists": bool(context),
    }


@app.delete("/memory/caller/{caller_id}")
async def clear_caller_memory(caller_id: str):
    """
    Clear all memory for a caller (GDPR compliance).
    This permanently removes all stored history for the specified caller.
    """
    memory_manager = await get_memory_manager()
    success = await memory_manager.clear_caller_memory(caller_id)
    
    return {
        "caller_id": caller_id,
        "cleared": success,
        "message": "All caller memory has been deleted." if success else "No memory found for this caller.",
    }


@app.get("/memory/status")
async def memory_status():
    """
    Get memory system status.
    Returns counts of active sessions and tracked callers.
    """
    memory_manager = await get_memory_manager()
    
    return {
        "status": "active",
        "active_sessions": len(memory_manager._sessions),
        "tracked_callers": len(memory_manager._caller_memories),
        "pattern_types": list(memory_manager._pattern_memory.keys()),
    }


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
            data = json.loads(f.read_text(encoding='utf-8'))
            reports.append({
                "filename": f.name,
                "id": f.name,
                "audio_file": data.get("audio_file", ""),
                "processed_at": data.get("processed_at", ""),
                "timestamp": data.get("processed_at", ""),  # Map for frontend compatibility
                "overall_risk": data.get("overall_risk", {}),
                "risk_score": data.get("overall_risk", {}).get("score", 0),  # Direct score for frontend
                "duration": data.get("duration_seconds"),
                "profanity_findings": data.get("profanity_findings", []),
                "regulatory_compliance": data.get("regulatory_compliance", {}),
                "obligation_analysis": data.get("obligation_analysis", []),
                "obligation_sentences": data.get("obligation_sentences", []),
                "emotion_analysis": data.get("emotion_analysis", {}),
                "segments": data.get("segments", []),
                "transcript": data.get("transcript", ""),
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

    return json.loads(filepath.read_text(encoding='utf-8'))


# ── WebSocket: Real-Time Analysis ──────────────────────────────────────────

@app.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time call analysis."""
    await handle_realtime_websocket(websocket)


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
