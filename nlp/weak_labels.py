# nlp/weak_labels.py
CAUSAL = [
    "caused", "leads to", "led to", "due to", "as a result",
    "because", "results in", "drives", "triggered"
]
CORREL = ["associated with", "linked to", "correlates with", "coincided with", "related to"]

def weak_label(text: str | None) -> str | None:
    if not text: return None
    t = text.lower()
    if any(k in t for k in CAUSAL): return "causal"
    if any(k in t for k in CORREL): return "correlational"
    return None
