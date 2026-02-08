#!/usr/bin/env python3
"""
Test script for the Financial Audio Intelligence API.
Usage:
  python test_api.py <audio_file_path>
  
Example:
  python test_api.py ../data/sample_calls/my_call.mp3
"""

import sys
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("\n1️⃣  Testing health endpoint...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {json.dumps(resp.json(), indent=2)[:200]}...")
    return resp.status_code == 200

def test_transcribe(audio_path: str):
    """Test Layer 1: Transcription only."""
    print(f"\n2️⃣  Testing Layer 1 (Transcription) with: {audio_path}")
    
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f)}
        resp = requests.post(f"{BASE_URL}/transcribe", files=files)
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✓ Status: {data['status']}")
        print(f"   ✓ Duration: {data.get('duration', 'N/A')}s")
        quality = data.get('audio_quality', {})
        print(f"   ✓ Audio quality: {quality.get('quality_label', quality.get('quality_score', 'N/A'))}")
        print(f"   ✓ Transcript (first 200 chars): {data['transcript'][:200]}...")
        return data
    else:
        print(f"   ✗ Error: {resp.status_code}")
        print(f"   {resp.text}")
        return None

def test_full_analysis(audio_path: str):
    """Test FULL PIPELINE: L1 + L2 + L3 + report."""
    print(f"\n3️⃣  Testing FULL PIPELINE with: {audio_path}")
    print("   ⏳ This will take 60-120 seconds (Backboard assistants running in parallel)...")
    
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f)}
        resp = requests.post(f"{BASE_URL}/analyze", files=files, timeout=300)
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n   ✓ Status: {data['status']}")
        print(f"   ✓ Processing time: {data['processing_time_seconds']}s")
        print(f"\n   📊 RESULTS:")
        print(f"      • Transcript length: {len(data['transcript'])} chars")
        print(f"      • PII found: {data['pii_count']} items")
        print(f"      • Text risk level: {data['text_risk_level']}")
        print(f"      • Overall risk: {data['overall_risk']['level']} (score: {data['overall_risk']['score']})")
        
        if data['profanity_findings']:
            print(f"      • Profanity findings: {len(data['profanity_findings'])} items")
        
        if isinstance(data['obligation_analysis'], dict):
            obligations = data['obligation_analysis'].get('obligations', [])
            print(f"      • Obligations detected: {len(obligations)}")
        
        if isinstance(data['intent_classification'], dict):
            intent = data['intent_classification'].get('primary_intent', 'unknown')
            print(f"      • Primary intent: {intent}")
        
        if isinstance(data['regulatory_compliance'], dict):
            comp_score = data['regulatory_compliance'].get('compliance_score')
            comp_status = data['regulatory_compliance'].get('overall_compliance')
            if comp_score is not None:
                print(f"      • Compliance score: {comp_score}/100")
            if comp_status:
                print(f"      • Compliance status: {comp_status}")
        
        print(f"\n   📁 Report saved to: {data['report_path']}")
        return data
    else:
        print(f"   ✗ Error: {resp.status_code}")
        print(f"   {resp.text}")
        return None

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n📝 Usage:")
        print("   1. Place an audio file in ../data/sample_calls/ or provide the path")
        print("   2. Run: python test_api.py <path_to_audio>")
        print("\n🔊 Supported formats: .mp3, .wav, .m4a, .mp4, .ogg, .flac, .webm")
        return
    
    audio_path = sys.argv[1]
    audio_file = Path(audio_path)
    
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Financial Audio Intelligence API - Test Suite")
    print(f"{'='*60}")
    print(f"Audio file: {audio_file.absolute()}")
    print(f"File size: {audio_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Test 1: Health
    if not test_health():
        print("\n❌ Server is not running. Start it with: python main.py")
        return
    
    # Test 2: Layer 1 only
    layer1_result = test_transcribe(audio_path)
    if not layer1_result:
        print("\n❌ Layer 1 (Transcription) failed")
        return
    
    # Test 3: Full pipeline
    full_result = test_full_analysis(audio_path)
    if full_result:
        print(f"\n✅ Full pipeline completed successfully!")
        print(f"\n{'='*60}")
    else:
        print(f"\n❌ Full pipeline failed")
    
if __name__ == "__main__":
    main()
