#!/usr/bin/env python3
"""Verify all new features in the improved report."""

import json
from pathlib import Path

report_path = Path(__file__).parent.parent / "data" / "outputs" / "report_20260209_033129.json"

with open(report_path, encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("NEW FEATURES VERIFICATION")
print("=" * 70)

# 1. Confidence scores
print("\n✓ OVERALL CONFIDENCE:", data.get("overall_confidence"))
print("\n✓ SEGMENT CONFIDENCE (first 3):")
for i, seg in enumerate(data["segments"][:3], 1):
    print(f"  Seg {i}: [{seg['start']:.1f}s-{seg['end']:.1f}s] "
          f"confidence={seg.get('confidence')} "
          f"speaker={seg.get('speaker')} "
          f"text={seg['text'][:50]}...")

# 2. Emotion & Stress per segment
print("\n✓ EMOTION/STRESS (first 5 segments):")
for i, seg in enumerate(data["segments"][:5], 1):
    print(f"  Seg {i}: emotion={seg.get('emotion', 'N/A')}, stress={seg.get('stress_level', 'N/A')}")

# 3. Emotion summary
print("\n✓ EMOTION ANALYSIS SUMMARY:")
ea = data.get("emotion_analysis", {})
print(f"  Overall stress: {ea.get('overall_stress')}")
print(f"  Stress distribution: {ea.get('stress_distribution')}")
print(f"  Segments analyzed: {ea.get('segments_analyzed')}")

# 4. Tamper detection
print("\n✓ TAMPER DETECTION:")
td = data.get("tamper_detection", {})
print(f"  Integrity score: {td.get('integrity_score')}")
print(f"  Tamper detected: {td.get('tamper_detected')}")
print(f"  Flags: {td.get('tampering_flags')}")
print(f"  Splice candidates: {len(td.get('splice_candidates', []))} points")
print(f"  Silence pattern: {td.get('silence_pattern')}")
print(f"  Replay: {td.get('replay_analysis')}")
print(f"  Spectral consistency: {td.get('spectral_consistency')}")

# 5. Speaker diarization quality
speakers = set(seg.get("speaker") for seg in data["segments"])
print(f"\n✓ SPEAKERS FOUND: {speakers}")
speaker_counts = {}
for seg in data["segments"]:
    sp = seg.get("speaker", "?")
    speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
print(f"  Speaker distribution: {speaker_counts}")

# 6. Unique segments with no_speech_prob
segs_with_nsp = [s for s in data["segments"] if "no_speech_prob" in s]
print(f"\n✓ SEGMENTS WITH no_speech_prob: {len(segs_with_nsp)}/{len(data['segments'])}")
if segs_with_nsp:
    print(f"  Sample: no_speech_prob={segs_with_nsp[0].get('no_speech_prob')}, "
          f"compression_ratio={segs_with_nsp[0].get('compression_ratio')}")

# 7. Financial entities (multilingual check)
print(f"\n✓ FINANCIAL ENTITIES: {len(data.get('financial_entities', []))}")
for e in data.get("financial_entities", [])[:5]:
    print(f"  {e['type']}: {e['value']}")

# 8. Obligation sentences
print(f"\n✓ OBLIGATION SENTENCES (L2): {len(data.get('obligation_sentences', []))}")
for o in data.get("obligation_sentences", [])[:3]:
    print(f"  Keywords: {o['keywords']}")
    print(f"  Sentence: {o['sentence'][:80]}...")

# 9. Overall risk
print(f"\n✓ OVERALL RISK:")
risk = data.get("overall_risk", {})
print(f"  Score: {risk.get('score')}")
print(f"  Level: {risk.get('level')}")
print(f"  Factors: {risk.get('risk_factors')}")

print("\n" + "=" * 70)
print(f"TOTAL SEGMENTS: {len(data['segments'])}")
print(f"REPORT KEYS: {list(data.keys())}")
print("=" * 70)
