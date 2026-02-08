"""
LAYER 1: Audio Forensics
- Groq Whisper large-v3 transcription (with segments)
- Librosa audio quality analysis (SNR, clipping, silence ratio)
- Speaker diarization via energy-based silence detection
- Segments merged: each entry has { start, end, speaker, text }
- Text normalization (cleanup filler words, fix whitespace)
- Multi-language support (auto-detect or specify)
"""

import os
import re
from pathlib import Path
from groq import Groq
import librosa
import numpy as np
import soundfile as sf


# ── Text Normalization ────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Clean up raw transcript text."""
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    # Fix common whisper artifacts
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)           # space before punctuation
    text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)    # ensure space after sentence-end
    # Remove repeated fillers (uh, um, erm) when they appear 3+ times in a row
    text = re.sub(r"(\b(?:uh|um|erm|ah)\b[,.]?\s*){3,}", "", text, flags=re.IGNORECASE)
    # Trim dangling whitespace again
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_segment_text(text: str) -> str:
    """Clean a single segment's text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


# ── Transcription ─────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str, groq_client: Groq, language: str | None = None) -> dict:
    """
    Transcribe audio using Groq Whisper large-v3.
    
    Args:
        audio_path: Path to audio file.
        groq_client: Initialized Groq client.
        language: ISO-639-1 language code (e.g. "en", "hi", "es").
                  If None, Whisper auto-detects the language.
    """
    with open(audio_path, "rb") as f:
        kwargs = {
            "file": (Path(audio_path).name, f.read()),
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        transcription = groq_client.audio.transcriptions.create(**kwargs)

    segments = []
    if hasattr(transcription, "segments") and transcription.segments:
        for s in transcription.segments:
            if isinstance(s, dict):
                seg_text = normalize_segment_text(s.get("text", ""))
                segments.append({
                    "start": round(s.get("start", 0), 2),
                    "end": round(s.get("end", 0), 2),
                    "text": seg_text,
                })
            else:
                seg_text = normalize_segment_text(getattr(s, "text", ""))
                segments.append({
                    "start": round(getattr(s, "start", 0), 2),
                    "end": round(getattr(s, "end", 0), 2),
                    "text": seg_text,
                })

    detected_lang = getattr(transcription, "language", language or "auto")

    return {
        "transcript": normalize_text(transcription.text),
        "language": detected_lang,
        "duration": getattr(transcription, "duration", None),
        "segments": segments,
    }


def analyze_audio_quality(audio_path: str) -> dict:
    """Analyze audio quality metrics using librosa."""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Signal-to-Noise Ratio (estimate)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms)
    snr_estimate = float(np.mean(rms_db) - np.min(rms_db))

    # Clipping detection
    clip_threshold = 0.99
    clipping_samples = int(np.sum(np.abs(y) > clip_threshold))
    clipping_ratio = clipping_samples / len(y)

    # Silence ratio (frames below -40dB)
    silence_threshold = -40  # dB
    silent_frames = int(np.sum(rms_db < silence_threshold))
    silence_ratio = silent_frames / len(rms_db) if len(rms_db) > 0 else 0

    # Average loudness
    avg_loudness = float(np.mean(rms_db))

    # Quality scoring
    quality_issues = []
    if snr_estimate < 10:
        quality_issues.append("Low signal-to-noise ratio")
    if clipping_ratio > 0.01:
        quality_issues.append("Audio clipping detected")
    if silence_ratio > 0.5:
        quality_issues.append("Excessive silence (>50%)")
    if avg_loudness < -35:
        quality_issues.append("Very low volume")

    quality_score = "good"
    if len(quality_issues) >= 3:
        quality_score = "poor"
    elif len(quality_issues) >= 1:
        quality_score = "fair"

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "snr_estimate_db": round(snr_estimate, 2),
        "clipping_ratio": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        "avg_loudness_db": round(avg_loudness, 2),
        "quality_score": quality_score,
        "quality_issues": quality_issues,
    }


def diarize_speakers(audio_path: str) -> list[dict]:
    """
    Basic speaker segmentation using energy-based silence detection.
    Labels alternating speech segments as Speaker A / Speaker B.
    (For production, replace with pyannote-audio or similar.)
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # Get RMS energy per frame
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Find speech vs silence
    threshold = np.mean(rms) * 0.3
    is_speech = rms > threshold

    # Group consecutive speech frames into segments
    segments = []
    in_speech = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            if end_time - start_time > 0.5:  # ignore segments < 0.5s
                segments.append({"start": round(start_time, 2), "end": round(end_time, 2)})
            in_speech = False

    # Handle last segment
    if in_speech:
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(len(is_speech), sr=sr, hop_length=hop_length)
        if end_time - start_time > 0.5:
            segments.append({"start": round(start_time, 2), "end": round(end_time, 2)})

    # Alternate speaker labels (heuristic)
    for i, seg in enumerate(segments):
        seg["speaker"] = "Agent" if i % 2 == 0 else "Customer"

    return segments


def _assign_speakers(segments: list[dict], diarization: list[dict]) -> list[dict]:
    """
    Merge speaker labels from diarization into transcript segments.
    Each segment gets a 'speaker' field based on which diarization
    turn overlaps it the most.
    """
    if not diarization:
        for seg in segments:
            seg["speaker"] = "Unknown"
        return segments

    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        best_speaker = "Unknown"
        best_overlap = 0

        for d in diarization:
            # Overlap between segment and diarization turn
            overlap_start = max(seg["start"], d["start"])
            overlap_end = min(seg["end"], d["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        # Fallback: if no overlap, use midpoint containment
        if best_speaker == "Unknown":
            for d in diarization:
                if d["start"] <= seg_mid <= d["end"]:
                    best_speaker = d["speaker"]
                    break

        seg["speaker"] = best_speaker

    return segments


def run_layer1(audio_path: str, groq_client: Groq, language: str | None = None) -> dict:
    """
    Run complete Layer 1 pipeline.
    Returns unified segments: each has { start, end, speaker, text }.
    
    Args:
        audio_path: Path to audio file.
        groq_client: Initialized Groq client.
        language: ISO-639-1 code or None for auto-detect.
    """
    transcript_result = transcribe_audio(audio_path, groq_client, language=language)
    quality_result = analyze_audio_quality(audio_path)
    diarization = diarize_speakers(audio_path)

    # Merge transcript segments + speaker diarization into one array
    merged_segments = _assign_speakers(transcript_result["segments"], diarization)

    return {
        "layer": "audio_forensics",
        "transcript": transcript_result["transcript"],
        "language": transcript_result["language"],
        "duration": transcript_result["duration"],
        "segments": merged_segments,
        "audio_quality": quality_result,
    }
