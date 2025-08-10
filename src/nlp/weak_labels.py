# nlp/weak_labels.py
CAUSAL = [
    # direct causation
    "caused", "causes", "causing", "leads to", "led to", "due to", "as a result",
    "because", "results in", "drives", "triggered", "brought about", "produced",
    "gave rise to", "responsible for", "resulted in", "therefore", "consequently",
    "hence", "provoked", "sparked", "induced", "generated", "stimulated",
    # probabilistic or contributory causation
    "contributed to", "plays a role in", "a factor in", "influences", "underlies",
    "underpinned", "prompted", "shaped", "sets off", "fuels"
]

CORREL = [
    "associated with", "linked to", "correlates with", "coincided with", "related to",
    "connected to", "in connection with", "tied to", "in conjunction with",
    "in parallel with", "goes along with", "matches with", "aligned with",
    "accompanies", "found alongside", "occurs with", "occurring with", "coupled with"
]

def weak_label(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    if any(k in t for k in CAUSAL):
        return "causal"
    if any(k in t for k in CORREL):
        return "correlational"
    return None

