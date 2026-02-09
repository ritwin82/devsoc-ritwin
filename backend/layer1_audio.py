"""
LAYER 1: Audio Forensics

- ElevenLabs Scribe (scribe_v2) transcription with diarization
- Librosa audio quality analysis (SNR, clipping, silence ratio)
- MFCC-based speaker diarization with agglomerative clustering
- Emotion & stress markers (pitch, energy, speech rate per segment)
- Tamper & replay detection (spectral discontinuity, silence anomalies)
- Text normalization and multi-language support
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

load_dotenv()


# ── FFmpeg helper: convert unsupported formats to WAV ─────────────────────

_FFMPEG_PATH: str | None = None

def _get_ffmpeg() -> str:
    """Resolve path to an FFmpeg binary (imageio-ffmpeg bundle or system)."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    # 1. Try imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        _FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG_PATH
    except Exception:
        pass
    # 2. Try system ffmpeg
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        _FFMPEG_PATH = p
        return _FFMPEG_PATH
    # 3. Check project-local ffmpeg.exe
    local = Path(__file__).parent / "ffmpeg.exe"
    if local.exists():
        _FFMPEG_PATH = str(local)
        return _FFMPEG_PATH
    raise FileNotFoundError(
        "FFmpeg not found. Install imageio-ffmpeg (`pip install imageio-ffmpeg`) "
        "or place ffmpeg.exe next to this file."
    )


def _ensure_wav(audio_path: str) -> tuple[str, bool]:
    """
    If *audio_path* is not a format librosa/soundfile can read natively
    (wav, mp3, flac, ogg), convert it to a temporary 16-bit PCM WAV using
    FFmpeg and return (wav_path, True). Otherwise return (audio_path, False).
    The caller must delete the temp file when *created* is True.
    """
    ext = Path(audio_path).suffix.lower()
    # Formats soundfile handles directly
    NATIVE = {".wav", ".mp3", ".flac", ".ogg"}
    if ext in NATIVE:
        return audio_path, False

    ffmpeg = _get_ffmpeg()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", audio_path, "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp.name],
            check=True,
            capture_output=True,
        )
        return tmp.name, True
    except subprocess.CalledProcessError as e:
        os.unlink(tmp.name)
        raise RuntimeError(
            f"FFmpeg conversion failed for {audio_path}: {e.stderr.decode(errors='replace')}"
        ) from e


# ── Financial Domain Vocabulary (Whisper prompt injection) ────────────────

FINANCIAL_VOCAB = {
    "en": (
        "account balance, interest rate, APR, EMI, KYC, NEFT, RTGS, SWIFT, "
        "mutual fund, SIP, debenture, collateral, mortgage, amortization, "
        "credit score, CIBIL, overdraft, fixed deposit, NPA, AML, forex, "
        "derivatives, futures, margin call, stop loss, lot size, leverage, "
        "spread, pip, drawdown, equity, dividend, NAV, compliance, "
        "disclosure, cooling-off period, penalty, processing fee, prepayment"
    ),
    "ru": (
        "процентная ставка, кредит, депозит, счёт, баланс, платёж, "
        "рассрочка, ипотека, залог, комиссия, штраф, пеня, вклад, "
        "инвестиция, портфель, маржа, лот, спред, позиция, сделка, "
        "брокер, трейдинг, акция, облигация, дивиденд, капитал, "
        "убыток, прибыль, просадка, стоп-лосс, тейк-профит"
    ),
    "hi": (
        "ब्याज दर, ऋण, जमा, खाता, शेष, भुगतान, किस्त, बंधक, "
        "गिरवी, कमीशन, जुर्माना, निवेश, पोर्टफोलियो, मार्जिन"
    ),
}


# ── Text Normalization ────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Clean up raw transcript text."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s*([A-ZА-ЯЁ])", r"\1 \2", text)
    text = re.sub(r"(\b(?:uh|um|erm|ah)\b[,.]?\s*){3,}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_segment_text(text: str) -> str:
    """Clean a single segment's text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


# ── Language code mapping (ISO 639-1 → ISO 639-3 for ElevenLabs) ──────────

LANGUAGE_CODE_MAP = {
    "en": "eng", "ru": "rus", "hi": "hin", "es": "spa", "fr": "fra",
    "de": "deu", "it": "ita", "pt": "por", "zh": "cmn", "ja": "jpn",
    "ko": "kor", "ar": "ara", "nl": "nld", "pl": "pol", "tr": "tur",
}


# ── Transcription with ElevenLabs Scribe ──────────────────────────────────

def transcribe_audio(audio_path: str, language: str | None = None) -> dict:
    """
    Transcribe with ElevenLabs Scribe (scribe_v2).
    - Uses native diarization for speaker identification
    - Tags audio events (laughter, applause, etc.)
    - Returns segments with start/end times
    """
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )

    # Convert language code to ISO 639-3 format
    lang_code = None
    if language:
        lang_code = LANGUAGE_CODE_MAP.get(language, language)
        # If already 3-letter code, use as-is
        if len(language) == 3:
            lang_code = language

    # Read audio file
    with open(audio_path, "rb") as f:
        audio_data = BytesIO(f.read())

    # Call ElevenLabs Speech-to-Text API
    transcription = elevenlabs_client.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v2",
        tag_audio_events=True,
        language_code=lang_code,  # None = auto-detect
        diarize=True,  # Enable speaker diarization
    )

    segments = []
    full_text_parts = []
    total_duration = 0.0

    # Known hallucination phrases to filter
    HALLUCINATION_PHRASES = {
        "и т.д.", "и т.д", "продолжение следует...", "продолжение следует",
        "субтитры сделал", "subtitles by", "subscribe", "подписывайтесь",
        "www.", "http", "...", "music", "♪", "thank you for watching",
    }

    # Process the response - ElevenLabs returns words with speaker info
    if hasattr(transcription, "words") and transcription.words:
        # Group words into segments by speaker and timing
        current_segment = None
        segment_words = []

        for word in transcription.words:
            word_text = getattr(word, "text", "") if hasattr(word, "text") else str(word.get("text", "") if isinstance(word, dict) else "")
            word_start = getattr(word, "start", 0) if hasattr(word, "start") else (word.get("start", 0) if isinstance(word, dict) else 0)
            word_end = getattr(word, "end", 0) if hasattr(word, "end") else (word.get("end", 0) if isinstance(word, dict) else 0)
            word_speaker = getattr(word, "speaker_id", None) if hasattr(word, "speaker_id") else (word.get("speaker_id", None) if isinstance(word, dict) else None)

            if word_end > total_duration:
                total_duration = word_end

            # Start new segment on speaker change or large gap (>1.5s)
            if current_segment is None:
                current_segment = {
                    "start": word_start,
                    "end": word_end,
                    "speaker": word_speaker,
                }
                segment_words = [word_text]
            elif word_speaker != current_segment["speaker"] or (word_start - current_segment["end"]) > 1.5:
                # Save current segment
                seg_text = normalize_segment_text(" ".join(segment_words))
                if seg_text.strip() and seg_text.strip().lower() not in HALLUCINATION_PHRASES:
                    segments.append({
                        "start": round(current_segment["start"], 2),
                        "end": round(current_segment["end"], 2),
                        "text": seg_text,
                        "confidence": None,  # ElevenLabs doesn't provide confidence scores
                        "speaker_id": current_segment["speaker"],
                    })
                    full_text_parts.append(seg_text)

                # Start new segment
                current_segment = {
                    "start": word_start,
                    "end": word_end,
                    "speaker": word_speaker,
                }
                segment_words = [word_text]
            else:
                # Continue current segment
                current_segment["end"] = word_end
                segment_words.append(word_text)

        # Don't forget the last segment
        if current_segment and segment_words:
            seg_text = normalize_segment_text(" ".join(segment_words))
            if seg_text.strip() and seg_text.strip().lower() not in HALLUCINATION_PHRASES:
                segments.append({
                    "start": round(current_segment["start"], 2),
                    "end": round(current_segment["end"], 2),
                    "text": seg_text,
                    "confidence": None,
                    "speaker_id": current_segment["speaker"],
                })
                full_text_parts.append(seg_text)

    # Fallback: if no words, try to get text directly
    elif hasattr(transcription, "text") and transcription.text:
        full_text = normalize_text(transcription.text)
        full_text_parts.append(full_text)
        segments.append({
            "start": 0,
            "end": total_duration,
            "text": full_text,
            "confidence": None,
        })

    # Detect language from response or use provided
    detected_lang = getattr(transcription, "language_code", None) or language or "auto"
    # Convert back to ISO 639-1 for consistency
    reverse_lang_map = {v: k for k, v in LANGUAGE_CODE_MAP.items()}
    if detected_lang in reverse_lang_map:
        detected_lang = reverse_lang_map[detected_lang]

    return {
        "transcript": normalize_text(" ".join(full_text_parts)),
        "language": detected_lang,
        "duration": round(total_duration, 2) if total_duration > 0 else None,
        "segments": segments,
        "overall_confidence": None,  # ElevenLabs doesn't provide confidence
    }


# ── Audio Quality ─────────────────────────────────────────────────────────

def analyze_audio_quality(audio_path: str) -> dict:
    """Analyze audio quality metrics using librosa."""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms)
    snr_estimate = float(np.mean(rms_db) - np.min(rms_db))

    clip_threshold = 0.99
    clipping_samples = int(np.sum(np.abs(y) > clip_threshold))
    clipping_ratio = clipping_samples / len(y)

    silence_threshold = -40
    silent_frames = int(np.sum(rms_db < silence_threshold))
    silence_ratio = silent_frames / len(rms_db) if len(rms_db) > 0 else 0

    avg_loudness = float(np.mean(rms_db))

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


# ── Speaker Diarization (MFCC + Spectral Clustering) ─────────────────────

def diarize_speakers(audio_path: str, n_speakers: int = 2) -> list[dict]:
    """
    Speaker diarization using MFCC + spectral feature extraction
    followed by agglomerative clustering.

    Extracts vocal embeddings (MFCCs, spectral centroid, bandwidth,
    rolloff, zero-crossing rate) per speech segment and clusters them
    to identify distinct speakers based on voice characteristics.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # Step 1: Voice Activity Detection via energy
    frame_length = int(0.025 * sr)   # 25 ms
    hop_length = int(0.010 * sr)     # 10 ms
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = np.mean(rms) * 0.3
    is_speech = rms > threshold

    # Group consecutive speech frames into segments
    raw_segments: list[dict] = []
    in_speech = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            if end_time - start_time > 0.3:
                raw_segments.append({"start": round(start_time, 2), "end": round(end_time, 2)})
            in_speech = False

    if in_speech:
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(len(is_speech), sr=sr, hop_length=hop_length)
        if end_time - start_time > 0.3:
            raw_segments.append({"start": round(start_time, 2), "end": round(end_time, 2)})

    if len(raw_segments) < 2:
        for seg in raw_segments:
            seg["speaker"] = "Speaker_1"
        return raw_segments

    # Step 2: Extract speaker embeddings per segment
    embeddings = []
    valid_indices = []

    for idx, seg in enumerate(raw_segments):
        start_sample = int(seg["start"] * sr)
        end_sample = min(int(seg["end"] * sr), len(y))
        segment_audio = y[start_sample:end_sample]

        if len(segment_audio) < frame_length * 2:
            continue

        try:
            # MFCC features (13 coefficients → mean + std = 26 features)
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Spectral features (4 additional features)
            sc = float(np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr)))
            sb = float(np.mean(librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)))
            sr_feat = float(np.mean(librosa.feature.spectral_rolloff(y=segment_audio, sr=sr)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=segment_audio)))

            # Combine into 30-dim embedding
            embedding = np.concatenate([mfcc_mean, mfcc_std, [sc, sb, sr_feat, zcr]])
            embeddings.append(embedding)
            valid_indices.append(idx)
        except Exception:
            continue

    if len(embeddings) < 2:
        for seg in raw_segments:
            seg["speaker"] = "Speaker_1"
        return raw_segments

    # Step 3: Normalize features & cluster
    embeddings_arr = np.array(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_arr)

    n_clusters = min(n_speakers, len(embeddings_scaled))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward",
    )
    labels = clustering.fit_predict(embeddings_scaled)

    # Step 4: Map clusters → speaker names
    # The cluster whose earliest speech segment comes first is "Agent"
    cluster_first_time: dict[int, float] = {}
    for i, label in enumerate(labels):
        seg_idx = valid_indices[i]
        t = raw_segments[seg_idx]["start"]
        if label not in cluster_first_time or t < cluster_first_time[label]:
            cluster_first_time[label] = t

    sorted_clusters = sorted(cluster_first_time, key=cluster_first_time.get)
    speaker_names = ["Agent", "Customer"] + [f"Speaker_{k+3}" for k in range(10)]
    label_map = {cid: speaker_names[i] for i, cid in enumerate(sorted_clusters)}

    label_dict = {valid_indices[i]: label_map[labels[i]] for i in range(len(labels))}

    for idx, seg in enumerate(raw_segments):
        seg["speaker"] = label_dict.get(idx, "Unknown")

    return raw_segments


# ── Emotion & Stress Analysis ────────────────────────────────────────────

def analyze_emotions(audio_path: str, segments: list[dict], sr_target: int = 22050) -> dict:
    """
    Analyze emotion/stress markers per transcript segment:
    - Pitch (F0) via pYIN: mean, std, range
    - Energy: RMS mean, variance
    - Speech rate: words / second
    - Derived labels: stress_level and emotion
    """
    y, sr = librosa.load(audio_path, sr=sr_target)

    segment_emotions: list[dict] = []
    stress_levels: list[str] = []

    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = min(int(seg["end"] * sr), len(y))
        segment_audio = y[start_sample:end_sample]
        duration = seg["end"] - seg["start"]

        # Skip very short segments
        if len(segment_audio) < int(0.15 * sr) or duration < 0.15:
            segment_emotions.append({
                "start": seg["start"],
                "end": seg["end"],
                "stress_level": "unknown",
                "emotion": "neutral",
            })
            continue

        # ── Pitch (F0) ──
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment_audio, fmin=50, fmax=500, sr=sr, frame_length=2048
            )
            f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:
            f0_valid = np.array([])

        pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0
        pitch_range = float(np.ptp(f0_valid)) if len(f0_valid) > 0 else 0.0

        # ── Energy ──
        rms = librosa.feature.rms(y=segment_audio)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # ── Speech rate (words/sec) ──
        word_count = len(seg.get("text", "").split())
        speech_rate = word_count / duration if duration > 0 else 0.0

        # ── Stress scoring ──
        stress_score = 0
        if pitch_std > 60:
            stress_score += 2
        elif pitch_std > 35:
            stress_score += 1
        if speech_rate > 4.0:
            stress_score += 2
        elif speech_rate > 3.0:
            stress_score += 1
        if energy_std > 0.04:
            stress_score += 1
        if pitch_range > 150:
            stress_score += 1

        stress_level = "high" if stress_score >= 4 else ("medium" if stress_score >= 2 else "low")

        # ── Emotion heuristic ──
        emotion = "neutral"
        if pitch_mean > 220 and energy_mean > 0.04:
            emotion = "agitated" if pitch_std > 50 else "excited"
        elif pitch_mean < 130 and energy_mean < 0.015:
            emotion = "calm"
        elif pitch_std > 50 and speech_rate > 3.5:
            emotion = "anxious"
        elif energy_mean > 0.06 and speech_rate > 3.5:
            emotion = "urgent"
        elif speech_rate < 1.5 and energy_mean < 0.02:
            emotion = "hesitant"

        stress_levels.append(stress_level)
        segment_emotions.append({
            "start": seg["start"],
            "end": seg["end"],
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_std_hz": round(pitch_std, 1),
            "pitch_range_hz": round(pitch_range, 1),
            "energy_mean": round(energy_mean, 4),
            "energy_std": round(energy_std, 4),
            "speech_rate_wps": round(speech_rate, 2),
            "stress_level": stress_level,
            "emotion": emotion,
        })

    # ── Aggregate summary ──
    stress_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for s in stress_levels:
        stress_counts[s] = stress_counts.get(s, 0) + 1

    overall_stress = "low"
    n = len(stress_levels) or 1
    if stress_counts["high"] > n * 0.3:
        overall_stress = "high"
    elif (stress_counts["medium"] + stress_counts["high"]) > n * 0.4:
        overall_stress = "medium"

    return {
        "segment_emotions": segment_emotions,
        "summary": {
            "overall_stress": overall_stress,
            "stress_distribution": stress_counts,
            "segments_analyzed": len(segment_emotions),
        },
    }


# ── Tamper & Replay Detection ────────────────────────────────────────────

def detect_tampering(audio_path: str) -> dict:
    """
    Analyze audio for signs of tampering / editing:
    1. Spectral discontinuity → splicing artifacts
    2. Silence pattern anomalies → editing indicators
    3. Replay detection → repeated audio chunks (MFCC cosine similarity)
    4. Encoding consistency → mixed audio sources (spectral rolloff variance)
    """
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    tampering_flags: list[str] = []
    hop_length = 512

    # ── 1. Spectral Discontinuity (splice detection) ──────────────────
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid_diff = np.abs(np.diff(spectral_centroid))
    # Use adaptive threshold: higher multiplier for low-SR phone audio
    sigma_mult = 5.0 if sr <= 16000 else 3.5
    threshold = np.mean(centroid_diff) + sigma_mult * np.std(centroid_diff)
    discontinuities = np.where(centroid_diff > threshold)[0]

    splice_points: list[float] = []
    for idx in discontinuities:
        t = round(float(librosa.frames_to_time(idx, sr=sr, hop_length=hop_length)), 2)
        splice_points.append(t)

    # Merge nearby splice points (within 0.5 s)
    merged_splices: list[float] = []
    for t in splice_points:
        if not merged_splices or t - merged_splices[-1] > 0.5:
            merged_splices.append(t)

    # Only flag if splice density exceeds reasonable threshold
    # (phone audio naturally has more discontinuities)
    splice_threshold = duration / 15 if sr <= 16000 else duration / 30
    if len(merged_splices) > splice_threshold:
        tampering_flags.append(
            f"Elevated spectral discontinuities: {len(merged_splices)} potential splice points"
        )

    # ── 2. Silence Pattern Analysis ───────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms)

    is_silent = rms_db < -45
    silence_durations: list[float] = []
    in_silence = False
    start_frame = 0

    for i, s in enumerate(is_silent):
        if s and not in_silence:
            start_frame = i
            in_silence = True
        elif not s and in_silence:
            dur = librosa.frames_to_time(i - start_frame, sr=sr, hop_length=hop_length)
            if dur > 0.05:
                silence_durations.append(round(float(dur), 3))
            in_silence = False

    if len(silence_durations) > 5:
        silence_std = float(np.std(silence_durations))
        silence_mean = float(np.mean(silence_durations))
        if silence_std < 0.03 and silence_mean > 0.3:
            tampering_flags.append(
                "Abnormally uniform silence gaps (possible post-editing)"
            )

    # ── 3. Replay Detection (MFCC cosine similarity) ──────────────────
    chunk_duration = 2.0
    chunk_size = int(chunk_duration * sr)
    n_chunks = len(y) // chunk_size
    replay_detected = False
    replay_pairs: list[dict] = []

    if n_chunks > 3:
        chunk_mfccs = []
        for i in range(min(n_chunks, 60)):
            chunk = y[i * chunk_size:(i + 1) * chunk_size]
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            chunk_mfccs.append(np.mean(mfcc, axis=1))

        from numpy.linalg import norm
        # Adaptive threshold: phone audio (<=16kHz) has narrower bandwidth
        # so chunks are naturally more similar — require near-identical match
        sim_threshold = 0.998 if sr <= 16000 else 0.97
        for i in range(len(chunk_mfccs)):
            for j in range(i + 3, len(chunk_mfccs)):
                a, b = chunk_mfccs[i], chunk_mfccs[j]
                n_a, n_b = norm(a), norm(b)
                if n_a > 0 and n_b > 0:
                    similarity = float(np.dot(a, b) / (n_a * n_b))
                    if similarity > sim_threshold:
                        replay_detected = True
                        replay_pairs.append({
                            "chunk_a_sec": round(i * chunk_duration, 1),
                            "chunk_b_sec": round(j * chunk_duration, 1),
                            "similarity": round(similarity, 4),
                        })

    # Only flag replay if a meaningful number of pairs found (not just noise)
    if replay_detected and len(replay_pairs) >= 3:
        tampering_flags.append(
            f"Potential audio replay: {len(replay_pairs)} highly similar non-adjacent segments"
        )

    # ── 4. Encoding Consistency ───────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)[0]
    rolloff_mean = float(np.mean(rolloff))
    rolloff_std = float(np.std(rolloff))
    consistency_ratio = rolloff_std / (rolloff_mean + 1e-10)

    # For phone-quality audio (<=16kHz), spectral envelope is naturally variable
    consistency_threshold = 0.7 if sr <= 16000 else 0.5
    if consistency_ratio > consistency_threshold:
        tampering_flags.append(
            "Inconsistent spectral envelope (possible mixed audio sources)"
        )

    # Weighted integrity: fewer flags = less penalty
    penalty_per_flag = 20
    integrity_score = max(0, 100 - len(tampering_flags) * penalty_per_flag)

    return {
        "integrity_score": integrity_score,
        "tamper_detected": len(tampering_flags) > 0,
        "tampering_flags": tampering_flags,
        "splice_candidates": merged_splices[:15],
        "silence_pattern": {
            "gap_count": len(silence_durations),
            "mean_duration_s": round(float(np.mean(silence_durations)), 3) if silence_durations else 0,
            "std_duration_s": round(float(np.std(silence_durations)), 3) if silence_durations else 0,
        },
        "replay_analysis": {
            "detected": replay_detected,
            "similar_pairs": replay_pairs[:5],
        },
        "spectral_consistency": round(consistency_ratio, 4),
    }


# ── Speaker Assignment (merge transcript segments + diarization) ─────────

def _assign_speakers(segments: list[dict], diarization: list[dict]) -> list[dict]:
    """
    Merge speaker labels from diarization into transcript segments.
    Each transcript segment gets a speaker based on maximum time overlap
    with a diarization turn.
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
            overlap_start = max(seg["start"], d["start"])
            overlap_end = min(seg["end"], d["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        # Fallback: midpoint containment
        if best_speaker == "Unknown":
            for d in diarization:
                if d["start"] <= seg_mid <= d["end"]:
                    best_speaker = d["speaker"]
                    break

        seg["speaker"] = best_speaker

    return segments


# ── Merge Emotion Labels into Segments ────────────────────────────────────

def _merge_emotions_into_segments(segments: list[dict], emotion_data: dict) -> list[dict]:
    """Embed emotion & stress labels directly into transcript segments."""
    seg_emotions = emotion_data.get("segment_emotions", [])

    for i, seg in enumerate(segments):
        if i < len(seg_emotions):
            em = seg_emotions[i]
            seg["stress_level"] = em.get("stress_level", "unknown")
            seg["emotion"] = em.get("emotion", "neutral")
        else:
            seg["stress_level"] = "unknown"
            seg["emotion"] = "neutral"

    return segments


# ── Main Layer 1 Pipeline ────────────────────────────────────────────────

def run_layer1(audio_path: str, language: str | None = None) -> dict:
    """
    Complete Layer 1 pipeline:
      1. Transcribe with ElevenLabs Scribe (scribe_v2)
      2. Audio quality analysis
      3. MFCC-based speaker diarization (clustering)
      4. Merge speakers into transcript segments
      5. Emotion & stress markers per segment
      6. Tamper & replay detection

    Returns unified segments with:
      { start, end, text, speaker, confidence, emotion, stress_level }
    """
    # ── Convert to WAV if needed (m4a, webm, mp4, etc.) ───────────────
    wav_path, was_converted = _ensure_wav(audio_path)
    # Use the converted WAV for librosa analysis.
    librosa_path = wav_path

    try:
        # 1. Transcription — ElevenLabs handles various formats
        transcript_result = transcribe_audio(audio_path, language=language)

        # 2. Audio quality (librosa)
        quality_result = analyze_audio_quality(librosa_path)

        # 3. Speaker diarization via MFCC clustering
        diarization = diarize_speakers(librosa_path)

        # 4. Merge speakers into transcript segments
        merged_segments = _assign_speakers(transcript_result["segments"], diarization)

        # 5. Emotion & stress
        emotion_result = analyze_emotions(librosa_path, merged_segments)

        # 6. Embed emotion labels into segments
        merged_segments = _merge_emotions_into_segments(merged_segments, emotion_result)

        # 7. Tamper detection
        tamper_result = detect_tampering(librosa_path)

        return {
            "layer": "audio_forensics",
            "transcript": transcript_result["transcript"],
            "language": transcript_result["language"],
            "duration": transcript_result["duration"],
            "segments": merged_segments,
            "audio_quality": quality_result,
            "overall_confidence": transcript_result["overall_confidence"],
            "emotion_analysis": emotion_result["summary"],
            "tamper_detection": tamper_result,
        }
    finally:
        # Clean up temp WAV file
        if was_converted and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass
