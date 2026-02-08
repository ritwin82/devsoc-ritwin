"""Quick test of faster-whisper transcription directly."""
import traceback
import sys

try:
    print("Importing...", flush=True)
    from faster_whisper import WhisperModel
    import time

    print("Loading model...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    t0 = time.time()
    model = WhisperModel("tiny", device="cpu", compute_type="auto")
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    audio_path = r"..\data\sample_calls\1735404531.458927.mp3"
    print(f"Transcribing: {audio_path}", flush=True)
    t0 = time.time()
    segments, info = model.transcribe(audio_path, language="en")

    print(f"Duration: {info.duration}s, Language: {info.language}", flush=True)
    count = 0
    for s in segments:
        if count < 10:
            print(f"  [{s.start:.1f}-{s.end:.1f}] {s.text.strip()[:80]}", flush=True)
        count += 1

    print(f"\nTotal segments: {count}", flush=True)
    print(f"Transcription took: {time.time()-t0:.1f}s", flush=True)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
