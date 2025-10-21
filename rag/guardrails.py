import re
PII = re.compile(r"(\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b\d{3}[- ]?\d{3}[- ]?\d{4}\b|\b[A-Z0-9]{17}\b)")
JAIL = re.compile(r"(ignore\s+previous\s+instructions|reveal\s+system\s+prompt|do\s+anything\s+now)", re.I)

def redact(s: str) -> str:
    return PII.sub("[REDACTED]", s)

def blocked(s: str) -> bool:
    return bool(JAIL.search(s))