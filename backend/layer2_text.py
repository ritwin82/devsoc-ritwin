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
# PII PATTERNS — Multilingual
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    # English / International
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
        r"\b(?:a/?c|account)\s*(?:no\.?|number|#)?\s*:?\s*\d{9,18}\b", re.IGNORECASE
    ),
    # Russian
    "phone_ru": re.compile(
        r"\+?7[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}"
    ),
    "passport_ru": re.compile(
        r"\b\d{4}\s?\d{6}\b"
    ),
    "inn_ru": re.compile(
        r"\b\d{10}(?:\d{2})?\b"
    ),
    "snils_ru": re.compile(
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{2}\b"
    ),
}

# ---------------------------------------------------------------------------
# FINANCIAL ENTITY PATTERNS
# ---------------------------------------------------------------------------
FINANCIAL_PATTERNS = {
    "currency_amount": re.compile(
        r"(?:(?:Rs\.?|INR|USD|\$|€|£|₹)\s*\d[\d,]*(?:\.\d{1,2})?)"
        r"|(?:\d[\d,]*(?:\.\d{1,2})?\s*(?:rupees|dollars|euros|pounds|lakhs?|crores?|thousand|hundred))",
        re.IGNORECASE,
    ),
    "currency_rub": re.compile(
        r"(?:\d[\d\s,]*(?:[.,]\d{1,2})?\s*(?:рублей|руб\.?|₽))"
        r"|(?:\d[\d\s,]*(?:[.,]\d{1,2})?\s*(?:тысяч|миллион(?:ов|а)?|млн)\s*(?:рублей|долларов|евро)?)"
        r"|(?:\d[\d\s,]*(?:[.,]\d{1,2})?\s*(?:долларов|евро))",
        re.IGNORECASE,
    ),
    "percentage": re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|per\s*cent|процент(?:ов|а)?)\b", re.IGNORECASE
    ),
    "date_reference": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        r"|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{2,4})"
        r"|(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})"
        r"|(?:\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*\d{2,4}?)",
        re.IGNORECASE,
    ),
    "loan_term": re.compile(
        r"\b\d+\s*(?:months?|years?|days?|EMI|installments?|месяц(?:ев|а)?|лет|год(?:а|ов)?|дней|дня)\b",
        re.IGNORECASE,
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
    # English
    "must", "shall", "required", "mandatory", "obligated",
    "need to", "have to", "should", "will be charged",
    "agree to", "consent", "acknowledge", "confirm",
    "i promise", "we guarantee", "committed to",
    "by signing", "terms and conditions", "cooling off",
    "within 30 days", "penalty", "fee", "interest rate",
    # Russian
    "должен", "обязан", "необходимо", "обещаю", "гарантирую",
    "подтверждаю", "согласен", "обязательно", "штраф", "комиссия",
    "процент", "условия", "договор", "контракт",
    "в течение", "обязуюсь", "ответственность",
    # Hindi
    "ज़रूरी", "अनिवार्य", "वादा", "सहमत", "शर्तें",
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
    """Extract sentences containing obligation keywords (word-boundary matching)."""
    doc = nlp(text)
    obligations = []

    for sent in doc.sents:
        sent_lower = sent.text.lower()
        matched_keywords = []
        for kw in OBLIGATION_KEYWORDS:
            # Use word-boundary regex to avoid substring matches
            # e.g. prevent "продолжение" from matching "должен"
            pattern = r'(?<!\w)' + re.escape(kw) + r'(?!\w)'
            if re.search(pattern, sent_lower):
                matched_keywords.append(kw)
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


def run_layer2(transcript: str, language: str = "en") -> dict:
    """Run complete Layer 2 pipeline on transcript text."""
    pii = detect_pii(transcript)
    financial_entities = extract_financial_entities(transcript)
    # spaCy NER — only run on English (en_core_web_sm)
    if language.lower().startswith("en"):
        spacy_entities = extract_spacy_entities(transcript)
    else:
        spacy_entities = []  # spaCy en model not applicable for non-English
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
