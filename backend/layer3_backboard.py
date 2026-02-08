"""
LAYER 3: Intelligence via Backboard.io

Three specialized assistants:
1. Obligation Detector  – finds commitments, promises, agreements
2. Intent Classifier    – classifies call intent/sentiment
3. Regulatory Checker   – checks compliance against uploaded policies (RAG)

Uses Backboard memory="Auto" so assistants learn patterns over time.
Policy documents are uploaded to the Regulatory Checker for RAG.
"""

import os
import asyncio
import time
from pathlib import Path
from backboard import BackboardClient

BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY", "")

# ── Assistant definitions ──────────────────────────────────────────────────

ASSISTANTS = {
    "obligation_detector": {
        "name": "Obligation Detector",
        "system_prompt": """You are a financial compliance analyst specializing in obligation detection.

Given a call transcript, identify ALL obligations, commitments, and promises made during the call.

For each obligation found, provide:
1. The exact quote from the transcript
2. Who made the obligation (agent or customer)
3. Type: payment_promise, service_commitment, deadline, penalty_warning, fee_disclosure, consent_given, follow_up_action
4. Severity: critical, important, informational
5. Whether it was properly disclosed/explained

Respond ONLY with valid JSON. The top-level keys must be "obligations" (array), "summary" (string), and "risk_flags" (array of strings).
Each item in the "obligations" array must have keys: "quote", "speaker", "type", "severity", "properly_disclosed" (boolean), and "notes".
Do not include any text outside the JSON object.""",
    },
    "intent_classifier": {
        "name": "Intent Classifier",
        "system_prompt": """You are an expert call analyst for financial services.

Given a call transcript, analyze and classify:

1. PRIMARY INTENT: What is the main purpose of this call?
   Categories: complaint, inquiry, sales_pitch, collection, support, verification, fraud_report, account_closure, loan_application, dispute

2. SENTIMENT ANALYSIS: Overall sentiment of each party
   - Agent sentiment: professional, aggressive, helpful, dismissive, neutral
   - Customer sentiment: satisfied, frustrated, angry, confused, neutral, anxious

3. CALL OUTCOME: How did the call end?
   - resolved, unresolved, escalated, callback_scheduled, transferred, dropped

4. KEY TOPICS discussed (list them)

5. COMPLIANCE TONE: Was the agent's tone appropriate for a regulated financial call?

Respond ONLY with valid JSON. The top-level keys must be: "primary_intent" (string), "secondary_intents" (array), "agent_sentiment" (string), "customer_sentiment" (string), "call_outcome" (string), "key_topics" (array), "compliance_tone" (string: appropriate, concerning, or violation), "tone_notes" (string), and "confidence" (number 0.0-1.0).
Do not include any text outside the JSON object.""",
    },
    "regulatory_checker": {
        "name": "Regulatory Compliance Checker",
        "system_prompt": """You are a regulatory compliance auditor for financial services calls.

You have access to uploaded policy documents containing banking regulations and prohibited phrases.

Given a call transcript along with pre-extracted data (PII findings, obligation sentences, profanity detected), perform a thorough compliance check:

1. REGULATION VIOLATIONS: Check against the uploaded policy documents. For each violation:
   - Which regulation was violated
   - The exact quote from the transcript
   - Severity: critical, major, minor
   - Recommended action

2. MANDATORY DISCLOSURES: Were all required disclosures made?
   - Recording consent
   - Interest rate disclosure
   - Fee disclosure
   - Cooling-off period mention (for new products)

3. PROHIBITED BEHAVIOR CHECK:
   - Threats or pressure tactics
   - Misleading statements
   - Unprofessional language
   - Calling outside permitted hours

4. DATA HANDLING: Were PII/sensitive data handled appropriately?

Respond ONLY with valid JSON. The top-level keys must be:
- "violations" (array of objects, each with keys: "regulation", "quote", "severity", "action_required")
- "disclosures" (object with boolean keys: "recording_consent", "interest_rate", "fees_disclosed", "cooling_off_period")
- "prohibited_behavior" (array of strings)
- "data_handling_issues" (array of strings)
- "overall_compliance" (string: compliant, needs_review, or non_compliant)
- "compliance_score" (integer 0-100)
- "recommendations" (array of strings)
Do not include any text outside the JSON object.""",
    },
}

# ── State: cached assistant/thread IDs ─────────────────────────────────────
_state = {
    "initialized": False,
    "assistant_ids": {},
    "thread_ids": {},
}


async def _get_client() -> BackboardClient:
    """Get Backboard client."""
    key = os.getenv("BACKBOARD_API_KEY", BACKBOARD_API_KEY)
    if not key:
        raise RuntimeError("BACKBOARD_API_KEY not set")
    return BackboardClient(api_key=key)


async def initialize_assistants():
    """Create all 3 assistants on Backboard and upload policy docs to the regulatory checker."""
    if _state["initialized"]:
        return _state

    client = await _get_client()

    for key, config in ASSISTANTS.items():
        # Create assistant
        assistant = await client.create_assistant(
            name=config["name"],
            system_prompt=config["system_prompt"],
        )
        _state["assistant_ids"][key] = assistant.assistant_id

        # Create a default thread per assistant
        thread = await client.create_thread(assistant.assistant_id)
        _state["thread_ids"][key] = thread.thread_id

    # Upload policy documents to the regulatory checker
    reg_assistant_id = _state["assistant_ids"]["regulatory_checker"]
    policy_dir = Path(__file__).parent.parent / "data" / "policies"
    if policy_dir.exists():
        for policy_file in policy_dir.iterdir():
            if policy_file.suffix in (".txt", ".ttx", ".pdf"):
                try:
                    doc = await client.upload_document_to_assistant(
                        reg_assistant_id,
                        str(policy_file),
                    )
                    # Wait for indexing (up to 30s)
                    for _ in range(15):
                        status = await client.get_document_status(doc.document_id)
                        if hasattr(status, "status"):
                            if status.status in ("indexed", "completed"):
                                break
                            if status.status == "failed":
                                print(f"WARNING: Doc {policy_file.name} indexing failed")
                                break
                        time.sleep(2)
                except Exception as e:
                    print(f"WARNING: Could not upload {policy_file.name}: {e}")

    _state["initialized"] = True
    return _state


async def _send_to_assistant(
    assistant_key: str,
    content: str,
    model_name: str = "gpt-4o",
    llm_provider: str = "openai",
) -> str:
    """Send a message to a specific assistant and get the response."""
    client = await _get_client()

    if not _state["initialized"]:
        await initialize_assistants()

    assistant_id = _state["assistant_ids"][assistant_key]

    # Create a fresh thread for each analysis to avoid context pollution
    thread = await client.create_thread(assistant_id)

    response = await client.add_message(
        thread_id=thread.thread_id,
        content=content,
        memory="Auto",
        stream=False,
        llm_provider=llm_provider,
        model_name=model_name,
    )

    return response.content


async def detect_obligations(transcript: str, layer2_data: dict) -> str:
    """Assistant 1: Detect obligations in the transcript."""
    prompt = f"""Analyze the following financial services call transcript for obligations and commitments.

TRANSCRIPT:
{transcript}

PRE-EXTRACTED OBLIGATION SENTENCES:
{_format_obligations(layer2_data.get('obligation_sentences', []))}

Provide your analysis in the specified JSON format."""

    return await _send_to_assistant("obligation_detector", prompt)


async def classify_intent(transcript: str) -> str:
    """Assistant 2: Classify call intent and sentiment."""
    prompt = f"""Analyze the following financial services call transcript.

TRANSCRIPT:
{transcript}

Provide your intent/sentiment analysis in the specified JSON format."""

    return await _send_to_assistant("intent_classifier", prompt)


async def check_regulatory_compliance(transcript: str, layer2_data: dict) -> str:
    """Assistant 3: Check regulatory compliance using RAG on uploaded policies."""
    prompt = f"""Perform a regulatory compliance audit on the following financial services call.

TRANSCRIPT:
{transcript}

PRE-EXTRACTED DATA:

PII Found ({layer2_data.get('pii_count', 0)} items):
{_format_pii(layer2_data.get('pii_detected', []))}

Profanity/Prohibited Phrases Found:
{_format_profanity(layer2_data.get('profanity_findings', []))}

Financial Entities Found:
{_format_financial(layer2_data.get('financial_entities', []))}

Check against the uploaded banking regulations and prohibited phrases documents.
Provide your compliance audit in the specified JSON format."""

    return await _send_to_assistant("regulatory_checker", prompt)


async def run_layer3(transcript: str, layer2_data: dict) -> dict:
    """Run all 3 Backboard assistants in parallel."""
    # Initialize if needed
    if not _state["initialized"]:
        await initialize_assistants()

    # Run all three assistants concurrently (don't crash if one fails)
    results = await asyncio.gather(
        detect_obligations(transcript, layer2_data),
        classify_intent(transcript),
        check_regulatory_compliance(transcript, layer2_data),
        return_exceptions=True,
    )

    obligation_result = results[0] if not isinstance(results[0], Exception) else f"Error: {results[0]}"
    intent_result = results[1] if not isinstance(results[1], Exception) else f"Error: {results[1]}"
    compliance_result = results[2] if not isinstance(results[2], Exception) else f"Error: {results[2]}"

    return {
        "layer": "backboard_intelligence",
        "obligation_analysis": _try_parse_json(obligation_result),
        "intent_classification": _try_parse_json(intent_result),
        "regulatory_compliance": _try_parse_json(compliance_result),
        "assistants_used": list(_state["assistant_ids"].keys()),
        "raw_responses": {
            "obligations": obligation_result,
            "intent": intent_result,
            "compliance": compliance_result,
        },
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_obligations(obligations: list) -> str:
    if not obligations:
        return "None detected locally."
    lines = []
    for ob in obligations:
        lines.append(f"- \"{ob['sentence']}\" (keywords: {', '.join(ob['keywords'])})")
    return "\n".join(lines)


def _format_pii(pii_list: list) -> str:
    if not pii_list:
        return "None detected."
    lines = []
    for p in pii_list:
        lines.append(f"- {p['type']}: {p['value'][:4]}*** (risk: {p['risk']})")
    return "\n".join(lines)


def _format_profanity(profanity_list: list) -> str:
    if not profanity_list:
        return "None detected."
    lines = []
    for p in profanity_list:
        lines.append(f"- [{p['type']}] \"{p['value']}\" (severity: {p['severity']})")
    return "\n".join(lines)


def _format_financial(entities: list) -> str:
    if not entities:
        return "None detected."
    lines = []
    for e in entities:
        lines.append(f"- {e['type']}: {e['value']}")
    return "\n".join(lines)


def _try_parse_json(text: str):
    """Try to parse JSON from the response, return raw text if fails."""
    import json
    # Try to find JSON block in the response
    try:
        # Look for ```json blocks
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        else:
            # Try parsing the whole thing
            return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return text
