"""
Real-Time Call Analysis via WebSocket

Flow:
1. Browser MediaRecorder captures audio chunks (5s intervals)
2. Chunks sent as base64 over WebSocket → decoded to WAV
3. Each chunk transcribed via Groq Whisper
4. Layer 2 text analysis runs on each new chunk (PII, profanity, obligations)
5. Periodically (every ~30s of accumulated text) run Layer 3 regulatory check
6. Results streamed back to frontend in real time
7. On stop → full audio saved → batch pipeline triggered
"""

import asyncio
import base64
import io
import json
import os
import tempfile
import wave
import struct
from datetime import datetime
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from groq import Groq
from dotenv import load_dotenv

from layer2_text import detect_pii, detect_profanity, extract_financial_entities, extract_obligations
from layer3_backboard import check_regulatory_compliance, initialize_assistants, _try_parse_json

load_dotenv()


class RealtimeSession:
    """Manages state for a single real-time call analysis session."""

    def __init__(self, language: str = "en"):
        self.language = language
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.transcript_chunks: list[dict] = []  # {timestamp, text}
        self.full_transcript = ""
        self.chunk_count = 0
        self.start_time = datetime.now()
        self.audio_buffers: list[bytes] = []  # raw PCM for post-call saving
        self.last_compliance_check_at = 0  # character count at last L3 check
        self.compliance_interval = 500  # run L3 every 500 chars of new text
        self.alerts: list[dict] = []

    def get_elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    async def process_audio_chunk(self, audio_b64: str) -> dict:
        """
        Process a single audio chunk:
        1. Decode base64 → save temp file
        2. Transcribe with Groq Whisper
        3. Run L2 checks on new text
        4. Optionally run L3 regulatory check
        """
        self.chunk_count += 1
        result = {
            "type": "chunk_result",
            "chunk_id": self.chunk_count,
            "timestamp": round(self.get_elapsed_seconds(), 1),
        }

        # Decode audio
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            result["error"] = f"Failed to decode audio: {e}"
            return result

        # Store raw audio for post-call batch
        self.audio_buffers.append(audio_bytes)

        # Save to temp file for Groq
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            # Transcribe with Groq Whisper
            transcript_text = await self._transcribe_chunk(temp_path)

            if transcript_text and transcript_text.strip():
                result["transcript"] = transcript_text
                self.transcript_chunks.append({
                    "timestamp": round(self.get_elapsed_seconds(), 1),
                    "text": transcript_text,
                    "chunk_id": self.chunk_count,
                })
                self.full_transcript += " " + transcript_text
                self.full_transcript = self.full_transcript.strip()

                # Run L2 real-time checks on the new chunk
                l2_alerts = self._run_l2_checks(transcript_text)
                if l2_alerts:
                    result["alerts"] = l2_alerts
                    self.alerts.extend(l2_alerts)

                # Check if we should run L3 regulatory check
                chars_since_last = len(self.full_transcript) - self.last_compliance_check_at
                if chars_since_last >= self.compliance_interval:
                    result["compliance_check_pending"] = True
            else:
                result["transcript"] = ""  # silence

            result["full_transcript"] = self.full_transcript
            result["total_chunks"] = self.chunk_count

        except Exception as e:
            result["error"] = f"Processing failed: {e}"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        return result

    async def _transcribe_chunk(self, audio_path: str) -> str:
        """Transcribe a single audio chunk using Groq Whisper."""
        try:
            with open(audio_path, "rb") as f:
                file_data = f.read()
            
            # Check minimum file size (skip near-empty chunks)
            if len(file_data) < 1000:
                return ""

            transcription = self.groq_client.audio.transcriptions.create(
                file=(Path(audio_path).name, file_data),
                model="whisper-large-v3",
                response_format="text",
                language=self.language if self.language != "auto" else None,
            )

            text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
            
            # Filter hallucination patterns
            hallucination_markers = {
                "thank you for watching", "subscribe", "like and subscribe",
                "subtitles by", "www.", "http", "...", "и т.д.",
                "продолжение следует", "подписывайтесь", "thanks for watching",
            }
            if text.lower().strip() in hallucination_markers:
                return ""
            if len(text) < 3:
                return ""

            return text
        except Exception as e:
            print(f"  Transcription error: {e}")
            return ""

    # ── Fraud / social-engineering keyword lists ────────────────────
    FRAUD_PHRASES = [
        "access to your computer", "access to ur computer",
        "remote access", "remote desktop", "teamviewer", "anydesk",
        "give me your password", "share your password", "tell me your pin",
        "send me money", "wire transfer now", "transfer funds immediately",
        "commit fraud", "commit a fraud", "launder money", "money laundering",
        "insider trading", "ponzi", "pyramid scheme",
        "gift card", "buy gift cards", "pay with gift cards",
        "do not tell anyone", "keep this secret", "don't tell your bank",
        "act now or else", "your account will be closed",
        "irs is suing you", "warrant for your arrest",
        "i am from the bank", "i am calling from microsoft",
        "social security number has been compromised",
        "verify your identity by sharing",
    ]
    FRAUD_KEYWORDS = [
        "fraud", "scam", "phishing", "extortion", "blackmail",
        "ransomware", "identity theft", "embezzlement",
    ]

    def _run_l2_checks(self, text: str) -> list[dict]:
        """Run Layer 2 checks on new text and return alerts."""
        alerts = []
        timestamp = round(self.get_elapsed_seconds(), 1)

        # PII detection
        pii_items = detect_pii(text)
        for pii in pii_items:
            alerts.append({
                "type": "pii",
                "severity": "high",
                "message": f"PII detected: {pii['type']} — {pii['value'][:4]}***",
                "timestamp": timestamp,
                "entity_type": pii["type"],
            })

        # Prohibited phrases / profanity
        profanity_items = detect_profanity(text)
        for p in profanity_items:
            alerts.append({
                "type": "prohibited" if p["type"] == "prohibited_phrase" else "profanity",
                "severity": p["severity"],
                "message": f"{'Prohibited phrase' if p['type'] == 'prohibited_phrase' else 'Profanity'}: \"{p['value']}\"",
                "timestamp": timestamp,
            })

        # Fraud / social-engineering detection
        text_lower = text.lower()
        matched_phrases = [p for p in self.FRAUD_PHRASES if p in text_lower]
        matched_keywords = [k for k in self.FRAUD_KEYWORDS if k in text_lower.split() or k in text_lower]
        if matched_phrases:
            alerts.append({
                "type": "social_engineering",
                "severity": "high",
                "message": f"Social-engineering phrase detected: \"{matched_phrases[0]}\"",
                "timestamp": timestamp,
                "keywords": matched_phrases,
            })
        if matched_keywords:
            alerts.append({
                "type": "fraud",
                "severity": "high",
                "message": f"Fraud keyword detected: {', '.join(matched_keywords)}",
                "timestamp": timestamp,
                "keywords": matched_keywords,
            })

        # Financial entities (informational)
        fin_entities = extract_financial_entities(text)
        for ent in fin_entities:
            alerts.append({
                "type": "financial_entity",
                "severity": "info",
                "message": f"{ent['type']}: {ent['value']}",
                "timestamp": timestamp,
            })

        # Obligation keywords
        obligations = extract_obligations(text)
        for ob in obligations:
            alerts.append({
                "type": "obligation",
                "severity": "medium",
                "message": f"Obligation detected: \"{ob['sentence'][:80]}...\"" if len(ob["sentence"]) > 80 else f"Obligation detected: \"{ob['sentence']}\"",
                "timestamp": timestamp,
                "keywords": ob["keywords"],
            })

        return alerts

    async def run_compliance_check(self) -> dict:
        """Run Layer 3 regulatory compliance check on accumulated transcript."""
        try:
            self.last_compliance_check_at = len(self.full_transcript)

            # Build lightweight layer2 data for the compliance check
            layer2_data = {
                "pii_detected": detect_pii(self.full_transcript),
                "pii_count": len(detect_pii(self.full_transcript)),
                "profanity_findings": detect_profanity(self.full_transcript),
                "financial_entities": extract_financial_entities(self.full_transcript),
            }

            raw_result = await check_regulatory_compliance(self.full_transcript, layer2_data)
            parsed = _try_parse_json(raw_result) if isinstance(raw_result, str) else raw_result

            return {
                "type": "compliance_update",
                "timestamp": round(self.get_elapsed_seconds(), 1),
                "data": parsed,
            }
        except Exception as e:
            return {
                "type": "compliance_update",
                "timestamp": round(self.get_elapsed_seconds(), 1),
                "error": str(e),
            }

    def save_full_audio(self) -> str | None:
        """Save all accumulated audio buffers as a single file for batch processing."""
        if not self.audio_buffers:
            return None

        upload_dir = Path("../data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_realtime_call.webm"
        filepath = upload_dir / filename

        try:
            # Concatenate all raw audio chunks
            with open(filepath, "wb") as f:
                for buf in self.audio_buffers:
                    f.write(buf)
            return str(filepath)
        except Exception as e:
            print(f"Failed to save audio: {e}")
            return None

    def get_session_summary(self) -> dict:
        """Get summary of the real-time session."""
        return {
            "duration_seconds": round(self.get_elapsed_seconds(), 1),
            "total_chunks": self.chunk_count,
            "transcript_length": len(self.full_transcript),
            "total_alerts": len(self.alerts),
            "alert_breakdown": {
                "pii": sum(1 for a in self.alerts if a["type"] == "pii"),
                "prohibited": sum(1 for a in self.alerts if a["type"] == "prohibited"),
                "profanity": sum(1 for a in self.alerts if a["type"] == "profanity"),
                "obligation": sum(1 for a in self.alerts if a["type"] == "obligation"),
                "financial_entity": sum(1 for a in self.alerts if a["type"] == "financial_entity"),
            },
        }


async def handle_realtime_websocket(websocket: WebSocket):
    """
    WebSocket handler for real-time call analysis.

    Protocol:
      Client → Server:
        { "type": "start", "language": "en" }
        { "type": "audio", "data": "<base64 audio>" }
        { "type": "stop" }

      Server → Client:
        { "type": "ready" }
        { "type": "chunk_result", ... }
        { "type": "compliance_update", ... }
        { "type": "session_ended", ... }
    """
    await websocket.accept()
    session: RealtimeSession | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg["type"] == "start":
                language = msg.get("language", "en")
                session = RealtimeSession(language=language)

                # Ensure Backboard assistants are initialized for compliance checks
                try:
                    await initialize_assistants()
                except Exception:
                    pass  # Will be initialized on first check

                await websocket.send_json({
                    "type": "ready",
                    "message": "Real-time analysis started",
                    "timestamp": session.get_elapsed_seconds(),
                })

            elif msg["type"] == "audio" and session:
                audio_data = msg.get("data", "")
                if not audio_data:
                    continue

                # Process the audio chunk
                result = await session.process_audio_chunk(audio_data)
                await websocket.send_json(result)

                # If compliance check is pending, run it asynchronously
                if result.get("compliance_check_pending"):
                    compliance = await session.run_compliance_check()
                    await websocket.send_json(compliance)

            elif msg["type"] == "stop" and session:
                # Save full audio for batch processing
                saved_path = session.save_full_audio()
                summary = session.get_session_summary()

                await websocket.send_json({
                    "type": "session_ended",
                    "summary": summary,
                    "full_transcript": session.full_transcript,
                    "all_alerts": session.alerts,
                    "saved_audio_path": saved_path,
                })

                # Trigger batch processing if audio was saved
                if saved_path:
                    await websocket.send_json({
                        "type": "batch_processing_started",
                        "message": "Full pipeline analysis started on recorded audio",
                        "audio_path": saved_path,
                    })

                    # Run full pipeline in background
                    try:
                        from pipeline import run_full_pipeline, save_report
                        report = await run_full_pipeline(saved_path, language=session.language)
                        report_path = await save_report(report)

                        await websocket.send_json({
                            "type": "batch_processing_complete",
                            "report_path": report_path,
                            "overall_risk": report.get("overall_risk", {}),
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "batch_processing_failed",
                            "error": str(e),
                        })

                break

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
