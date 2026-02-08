#!/usr/bin/env python3
"""Check if Russian text is properly stored in the JSON report."""

import json
from pathlib import Path

# Load the most recent report
report_path = Path(__file__).parent.parent / "data" / "outputs" / "report_20260209_024529.json"

with open(report_path, encoding="utf-8") as f:
    data = json.load(f)

print("=" * 60)
print("RUSSIAN TEXT VERIFICATION")
print("=" * 60)

print("\n✓ First 3 segments:")
for i, seg in enumerate(data["segments"][:3], 1):
    print(f"\n  Segment {i}: [{seg['start']:.1f}s-{seg['end']:.1f}s]")
    print(f"  Speaker: {seg['speaker']}")
    print(f"  Text: {seg['text'][:100]}...")

print("\n" + "=" * 60)
print("✓ Transcript sample (first 400 chars):")
print("=" * 60)
print(data["transcript"][:400])
print("\n" + "=" * 60)
