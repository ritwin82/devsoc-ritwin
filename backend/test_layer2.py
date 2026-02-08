"""Quick Layer 2 test - run after server is up."""
import requests
import json

BASE = "http://localhost:8000"

# Get transcript from Layer 1
print("Getting transcript from Layer 1...")
resp1 = requests.post(
    f"{BASE}/transcribe",
    files={"file": open("../data/sample_calls/1735404531.458927.mp3", "rb")},
    timeout=120,
)
transcript = resp1.json()["transcript"]
print(f"Transcript length: {len(transcript)} chars\n")

# Test Layer 2
print("=== LAYER 2: Text Processing ===")
resp2 = requests.post(f"{BASE}/analyze-text", json={"transcript": transcript}, timeout=60)
d = resp2.json()
print(f"HTTP: {resp2.status_code}")

if resp2.status_code == 200:
    print(f"\nPII detected: {d['pii_count']}")
    for p in d["pii_detected"][:5]:
        print(f"  - {p['type']}: {p['value'][:10]}...")

    print(f"\nFinancial entities: {len(d['financial_entities'])}")
    for e in d["financial_entities"][:5]:
        print(f"  - {e['type']}: {e['value']}")

    print(f"\nNamed entities: {len(d['named_entities'])}")
    for e in d["named_entities"][:8]:
        print(f"  - {e['label']}: {e['text']}")

    print(f"\nProfanity/Prohibited: {len(d['profanity_findings'])}")
    for p in d["profanity_findings"][:5]:
        print(f"  - [{p['type']}] \"{p['value']}\" (severity: {p['severity']})")

    print(f"\nObligation sentences: {len(d['obligation_sentences'])}")
    for o in d["obligation_sentences"][:5]:
        print(f"  - \"{o['sentence'][:100]}...\"")
        print(f"    keywords: {o['keywords']}")

    print(f"\nOverall risk level: {d['risk_level']}")
else:
    print(f"ERROR: {json.dumps(d, indent=2)}")
