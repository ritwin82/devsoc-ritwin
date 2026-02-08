import json

with open(r"C:\Users\Tarun\VIT\devsoc\data\outputs\report_20260209_023025.json") as f:
    d = json.load(f)

# Check top-level keys
print("Top-level keys:", list(d.keys()))
print()

# Check if diarization is gone
print("Has 'diarization' key?", "diarization" in d)
print()

# Show first 5 segments — should have start, end, speaker, text
print("=== FIRST 5 SEGMENTS ===")
for seg in d["segments"][:5]:
    print(json.dumps(seg, indent=2))
    print()

# Show last 3
print("=== LAST 3 SEGMENTS ===")
for seg in d["segments"][-3:]:
    print(json.dumps(seg, indent=2))
    print()

# Count speakers
speakers = set(s.get("speaker", "?") for s in d["segments"])
print(f"Speakers found: {speakers}")
print(f"Total segments: {len(d['segments'])}")

# Check transcript normalization
print(f"\nTranscript (first 300 chars):")
print(d["transcript"][:300])
