"""
LAYER 2: Basic Text Processing (Local)
- spaCy NER for financial entities (amounts, dates, orgs, persons)
- PII detection (phone, email, SSN, card numbers via regex)
- Profanity / prohibited phrase detection
- Obligation keyword extraction
"""

import re
import spacy
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# PII PATTERNS
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"
    ),
    "aadhaar": re.compile(
        r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"
    ),
    "pan": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b"
    ),
    "account_number": re.compile(
        r"\b\d{9,18}\b"
    ),
}

# ---------------------------------------------------------------------------
# FINANCIAL ENTITY PATTERNS
# ---------------------------------------------------------------------------
FINANCIAL_PATTERNS = {
    "currency_amount": re.compile(
        r"(?:(?:Rs\.?|INR|USD|\$|₹)\s*\d[\d,]*(?:\.\d{1,2})?)"
        r"|(?:\d[\d,]*(?:\.\d{1,2})?\s*(?:rupees|dollars|lakhs?|crores?|thousand|hundred))",
        re.IGNORECASE,
    ),
    "percentage": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|per\s*cent)\b", re.IGNORECASE
    ),
    "date_reference": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        r"|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{2,4})"
        r"|(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
        re.IGNORECASE,
    ),
    "loan_term": re.compile(
        r"\b\d+\s*(?:months?|years?|days?|EMI|installments?)\b", re.IGNORECASE
    ),
}

# ---------------------------------------------------------------------------
# PROFANITY / PROHIBITED PHRASES
# ---------------------------------------------------------------------------
PROFANITY_WORDS = {
    "damn", "hell", "shit", "fuck", "bastard", "ass", "crap",
    "idiot", "stupid", "dumb", "moron", "shut up",
}

PROHIBITED_PHRASES = [
    "we will take legal action immediately",
    "your credit score will be ruined",
    "we'll seize your assets",
    "you must decide right now",
    "this is your last chance",
    "don't tell anyone about this offer",
    "zero fees",
    "guaranteed approval",
    "risk-free investment",
]

# ---------------------------------------------------------------------------
# OBLIGATION KEYWORDS
# ---------------------------------------------------------------------------
OBLIGATION_KEYWORDS = [
    "must", "shall", "required", "mandatory", "obligated",
    "need to", "have to", "should", "will be charged",
    "agree to", "consent", "acknowledge", "confirm",
    "i promise", "we guarantee", "committed to",
    "by signing", "terms and conditions", "cooling off",
    "within 30 days", "penalty", "fee", "interest rate",
]


def detect_pii(text: str) -> list[dict]:
    """Detect PII entities using regex patterns."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "risk": "high",
            })
    return findings


def extract_financial_entities(text: str) -> list[dict]:
    """Extract financial entities (amounts, dates, percentages, loan terms)."""
    entities = []
    for ent_type, pattern in FINANCIAL_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({
                "type": ent_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })
    return entities


def extract_spacy_entities(text: str) -> list[dict]:
    """Extract named entities using spaCy NER."""
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "MONEY", "DATE", "CARDINAL", "PERCENT"}
    ]


def detect_profanity(text: str) -> list[dict]:
    """Detect profanity and prohibited phrases."""
    findings = []
    text_lower = text.lower()

    # Check prohibited phrases
    for phrase in PROHIBITED_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            findings.append({
                "type": "prohibited_phrase",
                "value": phrase,
                "start": idx,
                "severity": "high",
            })

    # Check profanity words
    words = re.findall(r"\b\w+\b", text_lower)
    for word in words:
        if word in PROFANITY_WORDS:
            findings.append({
                "type": "profanity",
                "value": word,
                "severity": "medium",
            })

    return findings


def extract_obligations(text: str) -> list[dict]:
    """Extract sentences containing obligation keywords."""
    doc = nlp(text)
    obligations = []
    text_lower = text.lower()

    for sent in doc.sents:
        sent_lower = sent.text.lower()
        matched_keywords = [kw for kw in OBLIGATION_KEYWORDS if kw in sent_lower]
        if matched_keywords:
            obligations.append({
                "sentence": sent.text.strip(),
                "keywords": matched_keywords,
                "start": sent.start_char,
                "end": sent.end_char,
            })

    return obligations


def load_policy_rules(policy_dir: str = "../data/policies") -> dict:
    """Load policy documents for reference."""
    rules = {}
    policy_path = Path(policy_dir)
    if policy_path.exists():
        for f in policy_path.iterdir():
            if f.suffix in (".txt", ".ttx"):
                rules[f.stem] = f.read_text(encoding="utf-8", errors="ignore")
    return rules


def run_layer2(transcript: str) -> dict:
    """Run complete Layer 2 pipeline on transcript text."""
    pii = detect_pii(transcript)
    financial_entities = extract_financial_entities(transcript)
    spacy_entities = extract_spacy_entities(transcript)
    profanity = detect_profanity(transcript)
    obligations = extract_obligations(transcript)

    # Risk summary
    risk_level = "low"
    if len(pii) > 0:
        risk_level = "medium"
    if any(p["severity"] == "high" for p in profanity) or len(pii) > 3:
        risk_level = "high"

    return {
        "layer": "text_processing",
        "pii_detected": pii,
        "pii_count": len(pii),
        "financial_entities": financial_entities,
        "named_entities": spacy_entities,
        "profanity_findings": profanity,
        "obligation_sentences": obligations,
        "risk_level": risk_level,
    }
