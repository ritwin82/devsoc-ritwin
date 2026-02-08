import json

with open(r"C:\Users\Tarun\VIT\devsoc\data\outputs\report_20260209_014114.json") as f:
    d = json.load(f)

print("=" * 60)
print("FULL REPORT - BACKBOARD ASSISTANT OUTPUTS")
print("=" * 60)

print("\n=== OBLIGATION ANALYSIS ===")
oa = d.get("obligation_analysis")
if isinstance(oa, dict):
    for ob in oa.get("obligations", []):
        print(f"\n  Speaker: {ob.get('speaker')}")
        print(f"  Type: {ob.get('type')}")
        print(f"  Severity: {ob.get('severity')}")
        print(f"  Quote: {ob.get('quote', '')[:120]}...")
        print(f"  Properly disclosed: {ob.get('properly_disclosed')}")
    print(f"\n  Summary: {oa.get('summary')}")
    print(f"  Risk flags: {oa.get('risk_flags')}")
else:
    print(f"  Raw: {str(oa)[:500]}")

print("\n=== INTENT CLASSIFICATION ===")
ic = d.get("intent_classification")
if isinstance(ic, dict):
    print(f"  Primary intent: {ic.get('primary_intent')}")
    print(f"  Secondary intents: {ic.get('secondary_intents')}")
    print(f"  Agent sentiment: {ic.get('agent_sentiment')}")
    print(f"  Customer sentiment: {ic.get('customer_sentiment')}")
    print(f"  Call outcome: {ic.get('call_outcome')}")
    print(f"  Key topics: {ic.get('key_topics')}")
    print(f"  Compliance tone: {ic.get('compliance_tone')}")
    print(f"  Tone notes: {ic.get('tone_notes')}")
    print(f"  Confidence: {ic.get('confidence')}")
else:
    print(f"  Raw: {str(ic)[:500]}")

print("\n=== REGULATORY COMPLIANCE ===")
rc = d.get("regulatory_compliance")
if isinstance(rc, dict):
    print(f"  Overall compliance: {rc.get('overall_compliance')}")
    print(f"  Compliance score: {rc.get('compliance_score')}/100")
    print(f"  Disclosures: {rc.get('disclosures')}")
    print(f"  Prohibited behavior: {rc.get('prohibited_behavior')}")
    print(f"  Data handling issues: {rc.get('data_handling_issues')}")
    for v in rc.get("violations", []):
        print(f"\n  VIOLATION:")
        print(f"    Regulation: {v.get('regulation')}")
        print(f"    Severity: {v.get('severity')}")
        print(f"    Quote: {v.get('quote', '')[:120]}")
        print(f"    Action: {v.get('action_required')}")
    print(f"\n  Recommendations: {rc.get('recommendations')}")
else:
    print(f"  Raw: {str(rc)[:500]}")

print("\n=== OVERALL RISK ===")
risk = d.get("overall_risk", {})
print(f"  Score: {risk.get('score')}/100")
print(f"  Level: {risk.get('level')}")
print(f"  Risk factors: {risk.get('risk_factors')}")
