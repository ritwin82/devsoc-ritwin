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
from layer5_data_export import (
    get_batch_processor, compute_analytics, compute_trends,
    export_all_reports_csv, MIN_BATCH_SIZE, EXPORTS_DIR
)
from layer6_querybot import (
    handle_query, get_suggested_questions, initialize_query_assistant
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Financial Audio Intelligence API",
    version="3.0.0",
    description="6-Layer compliance analysis for financial service call recordings with FinBERT-powered term extraction and NL query bot",
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
    try:
        await initialize_query_assistant()
    except Exception as e:
        print(f"⚠ Layer 6 Query Bot init deferred: {e}")


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
            "Layer 5: Data Export & Analytics (CSV + Batch)",
            "Layer 6: Query Bot (Natural Language Analytics)",
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
            str(dst), language,
            caller_id=caller_id,
            session_id=session_id
        )

        # Save report
        report_path = await save_report(report)
        report["report_path"] = report_path

        return report

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


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


# ── Layer 5: Batch Processing & CSV Export ─────────────────────────────────

@app.post("/batch/create")
async def create_batch():
    """
    Create a new batch for collecting multiple analyses.
    Minimum 6 files required to generate CSV.
    """
    processor = get_batch_processor()
    batch_id = processor.create_batch()
    
    return {
        "batch_id": batch_id,
        "status": "created",
        "min_files_required": MIN_BATCH_SIZE,
        "message": f"Add at least {MIN_BATCH_SIZE} files to generate CSV.",
    }


@app.post("/batch/{batch_id}/analyze")
async def batch_analyze(
    batch_id: str,
    file: UploadFile = File(...),
    language: str | None = None,
    caller_id: str | None = None,
):
    """
    Add a file to a batch and analyze it.
    Returns progress toward CSV generation threshold.
    """
    processor = get_batch_processor()
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp_str}_{file.filename}"
    dst = UPLOAD_DIR / safe_name
    
    try:
        with open(dst, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        
        # Run full pipeline
        report = await run_full_pipeline(
            str(dst), language,
            caller_id=caller_id,
            session_id=batch_id
        )
        
        # Save report
        await save_report(report)
        
        # Add to batch
        batch_status = processor.add_report(batch_id, report)
        
        return {
            "status": "success",
            "audio_file": safe_name,
            "batch_progress": batch_status,
            "report_summary": {
                "compliance_score": report.get("regulatory_compliance", {}).get("compliance_score"),
                "risk_level": report.get("overall_risk", {}).get("level"),
            },
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Batch analysis failed: {str(e)}")


@app.post("/batch/{batch_id}/finalize")
async def finalize_batch(batch_id: str):
    """
    Finalize a batch and generate CSV if threshold (6+ files) is met.
    """
    processor = get_batch_processor()
    result = processor.finalize_batch(batch_id)
    
    if "error" in result:
        raise HTTPException(400, result["error"])
    
    return result


@app.get("/batch/{batch_id}/status")
async def batch_status(batch_id: str):
    """Get status of a batch."""
    processor = get_batch_processor()
    result = processor.get_batch_status(batch_id)
    
    if "error" in result:
        raise HTTPException(404, result["error"])
    
    return result


@app.get("/export/csv")
async def export_csv():
    """
    Export all saved reports as CSV.
    Requires minimum 6 reports.
    """
    csv_path, error = export_all_reports_csv()
    
    if error:
        raise HTTPException(400, error)
    
    # Return file download
    from fastapi.responses import FileResponse
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=Path(csv_path).name
    )


@app.get("/analytics/summary")
async def analytics_summary():
    """
    Get aggregate analytics from all saved reports.
    Includes compliance scores, risk distributions, and totals.
    """
    return compute_analytics()


@app.get("/analytics/trends")
async def analytics_trends():
    """
    Get time-series data for visualization.
    Shows compliance and risk scores over time.
    """
    return compute_trends()


# ── Layer 6: Natural Language Query Bot ────────────────────────────────────

@app.post("/query")
async def query_bot(body: dict):
    """
    Layer 6: Ask questions about your call analysis data in natural language.
    
    Input: {"question": "How many calls have high risk?", "session_id": "optional-session-id"}
    
    Returns:
    - answer: Natural language response with data insights
    - data_context: Summary of data used to generate the answer
    """
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(400, "No question provided")
    
    session_id = body.get("session_id")
    
    try:
        result = await handle_query(question, session_id=session_id)
        return {
            "status": "success",
            **result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.get("/query/suggestions")
async def query_suggestions():
    """
    Get suggested questions for the query bot.
    Returns a list of example questions users can ask.
    """
    return {
        "suggestions": get_suggested_questions(),
    }


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
