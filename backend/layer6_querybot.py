"""
LAYER 6: Natural Language Query Bot

Provides a conversational interface for users to query call analysis data.
- Loads stats from CSV exports and JSON reports
- Uses a Backboard.io assistant for natural language understanding
- Returns formatted answers with data context

Dependencies: backboard-sdk, csv, json, pathlib
"""

import os
import csv
import json
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
from backboard import BackboardClient

BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY", "")

# Directories
EXPORTS_DIR = Path(__file__).parent.parent / "data" / "exports"
REPORTS_DIR = Path(__file__).parent.parent / "data" / "outputs"

# ── Query Bot Assistant ─────────────────────────────────────────────────────

QUERY_ASSISTANT_CONFIG = {
    "name": "Analytics Query Bot",
    "system_prompt": """You are an intelligent analytics assistant for a Financial Audio Intelligence platform.

Your job is to answer user questions about call analysis data in a clear, helpful, and conversational way.

You will be provided with STATISTICAL CONTEXT containing real data from analyzed calls. Use this data to answer questions accurately.

Guidelines:
1. Always base your answers on the provided data context — never make up numbers
2. If the data doesn't contain enough information to answer, say so clearly
3. Use natural, friendly language — avoid overly technical jargon
4. When citing numbers, be precise (use the exact values from the data)
5. For trends or comparisons, provide brief insight (e.g. "This is above average" or "Most calls fall in this category")
6. Format your responses with clear structure — use bullet points for lists, bold for key numbers
7. If asked about a specific call, reference it by its audio file name
8. You can answer questions about: compliance scores, risk levels, sentiments, intents, PII counts, durations, languages, emotions, stress levels, and more

Respond in plain text with markdown formatting. Keep answers concise but complete.""",
}

# ── State ───────────────────────────────────────────────────────────────────

_query_state = {
    "initialized": False,
    "assistant_id": None,
    "thread_ids": {},  # session_id -> thread_id
}


# ── Backboard Client ───────────────────────────────────────────────────────

async def _get_client() -> BackboardClient:
    """Get Backboard client."""
    key = os.getenv("BACKBOARD_API_KEY", BACKBOARD_API_KEY)
    if not key:
        raise RuntimeError("BACKBOARD_API_KEY not set")
    return BackboardClient(api_key=key)


async def initialize_query_assistant():
    """Create the query bot assistant on Backboard."""
    if _query_state["initialized"]:
        return

    client = await _get_client()

    assistant = await client.create_assistant(
        name=QUERY_ASSISTANT_CONFIG["name"],
        system_prompt=QUERY_ASSISTANT_CONFIG["system_prompt"],
    )
    _query_state["assistant_id"] = assistant.assistant_id
    _query_state["initialized"] = True
    print("✓ Layer 6 Query Bot assistant initialized")


# ── CSV Data Loading ────────────────────────────────────────────────────────

def _find_latest_csv() -> Optional[Path]:
    """Find the most recent CSV export file."""
    if not EXPORTS_DIR.exists():
        return None

    csv_files = sorted(EXPORTS_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    return csv_files[0] if csv_files else None


def load_csv_data() -> list[dict]:
    """Load data from the most recent CSV export."""
    csv_path = _find_latest_csv()
    if not csv_path:
        return []

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if row[key] == "":
                    continue
                try:
                    if "." in row[key]:
                        row[key] = float(row[key])
                    else:
                        row[key] = int(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)

    return rows


def load_report_data() -> list[dict]:
    """Load data from saved JSON reports (fallback if no CSV)."""
    if not REPORTS_DIR.exists():
        return []

    reports = []
    for f in sorted(REPORTS_DIR.glob("report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            reports.append(data)
        except Exception:
            continue

    return reports


# ── Statistics Computation ──────────────────────────────────────────────────

def _safe_avg(values: list) -> float:
    """Compute average of numeric values, ignoring None/empty."""
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else 0.0


def _distribution(values: list) -> dict:
    """Count occurrences of each unique value."""
    dist = {}
    for v in values:
        if v is not None and v != "":
            key = str(v)
            dist[key] = dist.get(key, 0) + 1
    return dist


def compute_data_context(rows: list[dict]) -> dict:
    """Compute comprehensive statistics from CSV/report data for the assistant."""
    if not rows:
        return {"status": "no_data", "message": "No analysis data available yet."}

    # Determine if CSV or report format
    is_csv = "report_id" in rows[0] if rows else False

    context = {
        "total_calls": len(rows),
        "data_source": "csv_export" if is_csv else "json_reports",
    }

    if is_csv:
        # ── CSV-based stats ──
        context["compliance_scores"] = {
            "average": _safe_avg([r.get("compliance_score") for r in rows]),
            "min": min((r.get("compliance_score", 100) for r in rows if isinstance(r.get("compliance_score"), (int, float))), default=0),
            "max": max((r.get("compliance_score", 0) for r in rows if isinstance(r.get("compliance_score"), (int, float))), default=0),
        }
        context["risk_scores"] = {
            "average": _safe_avg([r.get("overall_risk_score") for r in rows]),
            "min": min((r.get("overall_risk_score", 100) for r in rows if isinstance(r.get("overall_risk_score"), (int, float))), default=0),
            "max": max((r.get("overall_risk_score", 0) for r in rows if isinstance(r.get("overall_risk_score"), (int, float))), default=0),
        }
        context["risk_distribution"] = _distribution([r.get("overall_risk_level") for r in rows])
        context["compliance_distribution"] = _distribution([r.get("overall_compliance") for r in rows])
        context["intent_distribution"] = _distribution([r.get("primary_intent") for r in rows])
        context["sentiment"] = {
            "customer": _distribution([r.get("customer_sentiment") for r in rows]),
            "agent": _distribution([r.get("agent_sentiment") for r in rows]),
        }
        context["call_outcomes"] = _distribution([r.get("call_outcome") for r in rows])
        context["languages"] = _distribution([r.get("language") for r in rows])
        context["financial_tone"] = _distribution([r.get("financial_tone") for r in rows])

        context["totals"] = {
            "pii_detected": sum(r.get("pii_count", 0) for r in rows if isinstance(r.get("pii_count"), (int, float))),
            "profanity_found": sum(r.get("profanity_count", 0) for r in rows if isinstance(r.get("profanity_count"), (int, float))),
            "obligations": sum(r.get("obligation_count", 0) for r in rows if isinstance(r.get("obligation_count"), (int, float))),
            "violations": sum(r.get("violation_count", 0) for r in rows if isinstance(r.get("violation_count"), (int, float))),
            "financial_terms": sum(r.get("financial_term_count", 0) for r in rows if isinstance(r.get("financial_term_count"), (int, float))),
        }

        context["duration"] = {
            "total_seconds": sum(r.get("duration_seconds", 0) for r in rows if isinstance(r.get("duration_seconds"), (int, float))),
            "average_seconds": _safe_avg([r.get("duration_seconds") for r in rows]),
        }

        # Emotion stats
        context["emotions"] = {
            "avg_neutral_pct": _safe_avg([r.get("emotion_neutral_pct") for r in rows]),
            "avg_urgent_pct": _safe_avg([r.get("emotion_urgent_pct") for r in rows]),
            "avg_hesitant_pct": _safe_avg([r.get("emotion_hesitant_pct") for r in rows]),
            "avg_excited_pct": _safe_avg([r.get("emotion_excited_pct") for r in rows]),
        }

        context["stress"] = {
            "avg_low_pct": _safe_avg([r.get("stress_low_pct") for r in rows]),
            "avg_medium_pct": _safe_avg([r.get("stress_medium_pct") for r in rows]),
            "avg_high_pct": _safe_avg([r.get("stress_high_pct") for r in rows]),
        }

        # Per-call details
        context["calls"] = []
        for r in rows:
            context["calls"].append({
                "id": r.get("report_id", "unknown"),
                "audio_file": r.get("audio_file", "unknown"),
                "processed_at": r.get("processed_at", ""),
                "language": r.get("language", ""),
                "duration_seconds": r.get("duration_seconds", 0),
                "compliance_score": r.get("compliance_score", 0),
                "overall_compliance": r.get("overall_compliance", ""),
                "risk_score": r.get("overall_risk_score", 0),
                "risk_level": r.get("overall_risk_level", ""),
                "primary_intent": r.get("primary_intent", ""),
                "customer_sentiment": r.get("customer_sentiment", ""),
                "agent_sentiment": r.get("agent_sentiment", ""),
                "call_outcome": r.get("call_outcome", ""),
                "pii_count": r.get("pii_count", 0),
                "violation_count": r.get("violation_count", 0),
            })

    else:
        # ── JSON report-based stats (fallback) ──
        context["compliance_scores"] = {
            "average": _safe_avg([
                r.get("regulatory_compliance", {}).get("compliance_score")
                for r in rows if isinstance(r.get("regulatory_compliance"), dict)
            ]),
        }
        context["risk_scores"] = {
            "average": _safe_avg([
                r.get("overall_risk", {}).get("score")
                for r in rows if isinstance(r.get("overall_risk"), dict)
            ]),
        }

        context["calls"] = []
        for r in rows[:20]:  # Limit to 20 for context window
            context["calls"].append({
                "audio_file": r.get("audio_file", "unknown"),
                "processed_at": r.get("processed_at", ""),
                "language": r.get("language", ""),
                "duration_seconds": r.get("duration_seconds", 0),
                "compliance_score": r.get("regulatory_compliance", {}).get("compliance_score") if isinstance(r.get("regulatory_compliance"), dict) else None,
                "risk_score": r.get("overall_risk", {}).get("score") if isinstance(r.get("overall_risk"), dict) else None,
                "risk_level": r.get("overall_risk", {}).get("level") if isinstance(r.get("overall_risk"), dict) else None,
            })

    return context


# ── Query Handler ───────────────────────────────────────────────────────────

async def handle_query(question: str, session_id: Optional[str] = None) -> dict:
    """
    Handle a natural language query about call analysis data.

    Args:
        question: The user's question in natural language
        session_id: Optional session ID for conversation continuity

    Returns:
        Dictionary with 'answer' and 'data_context'
    """
    # Initialize if needed
    if not _query_state["initialized"]:
        await initialize_query_assistant()

    # Load data and compute context
    csv_rows = load_csv_data()
    if csv_rows:
        data_context = compute_data_context(csv_rows)
    else:
        report_rows = load_report_data()
        data_context = compute_data_context(report_rows)

    # Build the enriched prompt
    context_json = json.dumps(data_context, indent=2, default=str)

    prompt = f"""STATISTICAL CONTEXT (real data from analyzed calls):
{context_json}

USER QUESTION: {question}

Answer the question using the data above. Be precise with numbers and provide helpful insights."""

    # Get or create thread for this session
    client = await _get_client()
    assistant_id = _query_state["assistant_id"]

    if session_id and session_id in _query_state["thread_ids"]:
        thread_id = _query_state["thread_ids"][session_id]
    else:
        thread = await client.create_thread(assistant_id)
        thread_id = thread.thread_id
        if session_id:
            _query_state["thread_ids"][session_id] = thread_id

    # Send query to assistant
    response = await client.add_message(
        thread_id=thread_id,
        content=prompt,
        memory="Auto",
        stream=False,
        llm_provider="openai",
        model_name="gpt-4o",
    )

    return {
        "answer": response.content,
        "data_context": {
            "total_calls": data_context.get("total_calls", 0),
            "data_source": data_context.get("data_source", "none"),
        },
        "session_id": session_id,
    }


# ── Suggested Questions ─────────────────────────────────────────────────────

def get_suggested_questions() -> list[dict]:
    """Return a list of suggested questions for the query bot."""
    return [
        {
            "text": "How many calls have been analyzed so far?",
            "category": "overview",
        },
        {
            "text": "What is the average compliance score across all calls?",
            "category": "compliance",
        },
        {
            "text": "Which calls have high risk levels?",
            "category": "risk",
        },
        {
            "text": "What are the most common customer sentiments?",
            "category": "sentiment",
        },
        {
            "text": "How many PII items were detected in total?",
            "category": "privacy",
        },
        {
            "text": "What is the breakdown of call intents?",
            "category": "intent",
        },
        {
            "text": "Which calls had the worst compliance scores?",
            "category": "compliance",
        },
        {
            "text": "What is the average call duration?",
            "category": "overview",
        },
        {
            "text": "Show me the stress level distribution across calls",
            "category": "emotion",
        },
        {
            "text": "Are there any patterns in call outcomes?",
            "category": "outcome",
        },
    ]
