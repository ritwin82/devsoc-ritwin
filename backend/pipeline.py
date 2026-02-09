"""
Pipeline Orchestrator
Chains Layer 1 → Layer 2 → Layer 3 and produces final report.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from layer1_audio import run_layer1
from layer2_text import run_layer2
from layer3_backboard import run_layer3, initialize_assistants


async def run_full_pipeline(
    audio_path: str, 
    groq_client: Groq, 
    language: str | None = None,
    caller_id: str | None = None,
    session_id: str | None = None
) -> dict:
    """
    Execute the full 3-layer analysis pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        groq_client: Groq client for transcription
        language: Optional language hint
        caller_id: Optional caller identifier for memory persistence
        session_id: Optional session ID for grouping related analyses
    
    Layers:
        Layer 1: Audio Forensics  (Groq Whisper + Librosa)
        Layer 2: Text Processing  (spaCy + Regex + PII)
        Layer 3: Intelligence     (Backboard assistants + Memory)
    """
    start_time = datetime.now()

    # ── Layer 1: Audio Forensics ──────────────────────────────────────
    layer1 = run_layer1(audio_path, language=language)
    transcript = layer1["transcript"]

    if not transcript or not transcript.strip():
        return {
            "status": "error",
            "error": "Transcription returned empty — audio may be silent or corrupted.",
            "audio_quality": layer1.get("audio_quality"),
        }

    # ── Layer 2: Text Processing ──────────────────────────────────────
    detected_lang = layer1.get("language", "en")
    layer2 = run_layer2(transcript, language=detected_lang)

    # ── Layer 3: Backboard Intelligence (with memory) ─────────────────
    layer3 = await run_layer3(
        transcript, 
        layer2,
        session_id=session_id,
        caller_id=caller_id
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # ── Assemble final report ─────────────────────────────────────────
    report = {
        "status": "success",
        "processed_at": start_time.isoformat(),
        "processing_time_seconds": round(elapsed, 2),
        "audio_file": Path(audio_path).name,
        
        # Memory context
        "session_id": layer3.get("session_id"),
        "caller_id": layer3.get("caller_id"),

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

    return report


def _compute_overall_risk(layer1: dict, layer2: dict, layer3: dict) -> dict:
    """Compute overall risk score from all layers."""
    risk_factors = []
    score = 100  # Start at 100, deduct for issues

    # ── Tamper risk (Layer 1) ──
    tamper = layer1.get("tamper_detection", {})
    if isinstance(tamper, dict) and tamper.get("tamper_detected"):
        flags = tamper.get("tampering_flags", [])
        risk_factors.append(f"Audio integrity concern: {len(flags)} tampering indicator(s)")
        score -= len(flags) * 10

    # ── Confidence risk (Layer 1) ──
    confidence = layer1.get("overall_confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.5:
        risk_factors.append(f"Low transcription confidence ({confidence:.1%})")
        score -= 10

    # ── Emotion / stress risk (Layer 1) ──
    emotion = layer1.get("emotion_analysis", {})
    if isinstance(emotion, dict) and emotion.get("overall_stress") == "high":
        risk_factors.append("High stress detected in call")
        score -= 5

    # ── PII risk (Layer 2) ──
    pii_count = layer2.get("pii_count", 0)
    if pii_count > 0:
        risk_factors.append(f"{pii_count} PII item(s) exposed in conversation")
        score -= min(pii_count * 5, 20)

    # ── Profanity risk (Layer 2) ──
    profanity = layer2.get("profanity_findings", [])
    high_severity = [p for p in profanity if p.get("severity") == "high"]
    if high_severity:
        risk_factors.append(f"{len(high_severity)} prohibited phrase(s) detected")
        score -= len(high_severity) * 15

    if any(p.get("type") == "profanity" for p in profanity):
        risk_factors.append("Profanity used in call")
        score -= 10

    # ── Compliance from Layer 3 ──
    compliance = layer3.get("regulatory_compliance", {})
    if isinstance(compliance, dict):
        violations = compliance.get("violations", [])
        critical = [v for v in violations if v.get("severity") == "critical"]
        if critical:
            risk_factors.append(f"{len(critical)} critical regulation violation(s)")
            score -= len(critical) * 20

        comp_score = compliance.get("compliance_score")
        if isinstance(comp_score, (int, float)):
            score = int(score * 0.4 + comp_score * 0.6)

    score = max(score, 0)

    if score >= 80:
        level = "low"
    elif score >= 50:
        level = "medium"
    else:
        level = "high"

    return {
        "score": score,
        "level": level,
        "risk_factors": risk_factors,
    }


async def save_report(report: dict, output_dir: str = "../data/outputs") -> str:
    """Save report as JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = out / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    return str(filepath)
