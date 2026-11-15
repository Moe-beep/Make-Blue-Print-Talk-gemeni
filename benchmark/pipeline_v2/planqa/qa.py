"""Question answering on the plan graph, with evidence and explanation.

route(question)          -> intent (en / fi / de keywords; order matters, see INTENTS)
answer(graph, intent)    -> Answer: value, confidence, evidence ids, explanation

Evidence ids point to rooms (r..), texts (t..) and symbols (o..) of the graph, so
the report can highlight exactly what was counted. Without any evidence (e.g.
layout model not available) the answer is the most frequent answer of that
question type in the dataset, clearly marked as fallback.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .schemas import GraphRoom, PlanGraph, TextKind

# Order matters: e.g. the sink question mentions "kitchen and bathroom", the bathroom
# question mentions "separate WC rooms", Finnish 'kylpy' is in bathtub and bathroom.
INTENTS: list[tuple[str, str]] = [
    ("floors", r"\bfloors\b|storeys|stories|kerrosta\b|stockwerk|geschoss|etagen"),
    ("windows", r"window|ikkun|fenster"),
    ("doors", r"\bdoors?\b|\bovea\b|ovien|\btür"),
    ("showers_bathtubs", r"shower|bathtub|suihk|kylpyamme|dusche|badewanne"),
    ("sinks", r"\bsinks?\b|wash ?basin|allas|altaa|waschbecken|spüle"),
    ("toilets", r"toilet bowl|wc seat|wc-istui|toilettensitz|wc-sitz|wie viele toiletten"),
    ("saunas", r"sauna"),
    ("bathrooms", r"bathroom|washroom|kylpy|pesuhuone|badezimmer|bäder"),
    ("wc_rooms", r"wc rooms|toilet rooms|wc-tila|wc-räume|wc-raum"),
    ("kitchens", r"kitchen|keittiö|küche"),
    ("bedrooms", r"bedroom|makuuhuone|schlafzimmer"),
    ("balconies", r"balcon|parveke|balkon"),
]

# Most frequent answer per type in questions.jsonl (CubiCasa part) - used only without evidence
PRIOR = {"floors": 1, "windows": 4, "doors": 6, "showers_bathtubs": 1, "sinks": 2, "toilets": 1, "saunas": 1,
         "bathrooms": 1, "wc_rooms": 0, "kitchens": 1, "bedrooms": 1, "balconies": 0}

# Room labels as written in the plans (the questions themselves name these abbreviations).
# A label part matches one category; compound labels like 'OH+K' are split.
LABELS = {
    "bathroom": r"^(KH|KPH|PH|PE|PESUH|PESUHUONE|SH|KYLPYH\w*|BAD|BATH\w*)$",
    "wc": r"^(WC|INVA-?WC|WC\d)$",
    "sauna": r"^(S|SAUNA|LH|LÖYLYH\w*)$",
    "kitchen": r"^(K|KK|KT|TK|AK|KEITTIÖ|KEITTOKOMERO|AVOK\w*|TUPAK\w*|KÜCHE|KITCHEN\w*)$",
    "bedroom": r"^(MH\d*|MAKUUH\w*|SCHLAFEN|BED\w*)$",
    "balcony": r"^(P|PARV\w*|LASIT\w*|BALKON|BALCONY)$",
    "terrace": r"^(TERASSI|TER|TERRASSE|TERRACE|PATIO)$",
}
MODEL_CLASS = {"Bath": "bathroom", "Kitchen": "kitchen", "Bedroom": "bedroom", "Outdoor": "balcony"}
OBJECT_OF = {"windows": "window", "doors": "door", "toilets": "toilet", "sinks": "sink"}
FLOOR_LABEL = re.compile(r"(\d\.?\s*(krs|kerros|og|og\.|stock))|kellari|ullakko|erdgeschoss|dachgeschoss", re.I)


@dataclass
class Answer:
    intent: Optional[str]
    value: Optional[float]
    unit: str = "count"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""
    fallback: bool = False

    def dict(self):
        return asdict(self)


def route(question: str) -> Optional[str]:
    q = question.lower()
    for intent, rx in INTENTS:
        if re.search(rx, q):
            return intent
    return None


def label_categories(label: Optional[str]) -> set[str]:
    if not label:
        return set()
    parts = [p for p in re.split(r"[+/,&\s]+", label.upper().strip(" .:")) if p]
    return {cat for p in parts for cat, rx in LABELS.items() if re.match(rx, p)}


def room_categories(room: GraphRoom) -> set[str]:
    """Label first (the drawing says what the room is), layout-model class second."""
    from_label = label_categories(room.label_text) - {"terrace"}
    if from_label:
        return from_label
    if room.room_type.value == "sauna":
        return {"sauna"}
    if room.model_class == "Outdoor" and "terrace" in label_categories(room.label_text):
        return set()
    cat = MODEL_CLASS.get(room.model_class)
    return {cat} if cat else set()


def answer(graph: PlanGraph, intent: Optional[str]) -> Answer:
    if intent is None:
        return Answer(None, None, explanation="Fragetyp nicht unterstützt.", fallback=True)
    has_layout = bool(graph.rooms or graph.objects)
    if intent in OBJECT_OF:
        return _count_objects(graph, intent, has_layout)
    if intent == "floors":
        return _count_floors(graph, has_layout)
    if intent == "showers_bathtubs":
        return _count_showers_bathtubs(graph, has_layout)
    return _count_rooms(graph, intent)


# ---------------------------------------------------------------- answer types

def _prior(intent: str, why: str) -> Answer:
    return Answer(intent, PRIOR[intent], confidence=0.2, fallback=True,
                  explanation=f"{why} Standardantwort = häufigste Antwort dieses Fragetyps ({PRIOR[intent]}).")


def _count_objects(graph: PlanGraph, intent: str, has_layout: bool) -> Answer:
    kind = OBJECT_OF[intent]
    objs = [o for o in graph.objects if o.kind == kind]
    if not has_layout:
        return _prior(intent, "Layout-Modell nicht gelaufen, keine Symbole erkannt.")
    conf = sum(o.conf for o in objs) / len(objs) if objs else 0.5
    return Answer(intent, len(objs), confidence=round(conf, 2), evidence=[o.id for o in objs],
                  explanation=f"{len(objs)} Symbol(e) '{kind}' vom Layout-Modell erkannt"
                              + (f" (mittlere Wahrscheinlichkeit {conf:.2f})." if objs else "."))


def _count_floors(graph: PlanGraph, has_layout: bool) -> Answer:
    labels = [t for t in graph.texts if FLOOR_LABEL.search(t.text)]
    n_outlines = len(graph.floors)
    if len(labels) >= 2 and len(labels) >= n_outlines:
        return Answer("floors", len(labels), confidence=0.7, evidence=[t.id for t in labels],
                      explanation=f"{len(labels)} Geschossbezeichnungen im Plan gelesen: "
                                  + ", ".join(f"„{t.text}“" for t in labels) + ".")
    if not has_layout:
        return _prior("floors", "Kein Layout-Modell, keine Geschossbezeichnungen gefunden.")
    n = max(1, n_outlines)
    return Answer("floors", n, confidence=0.8 if n == 1 else 0.6,
                  explanation=f"{n_outlines} getrennte Gebäudeumriss(e) im Layout erkannt.")


def _rooms_of(graph: PlanGraph, cat: str) -> tuple[list[GraphRoom], list]:
    """Rooms of a category + labels of that category that lie outside every room
    (layout model missed the room but the drawing names it)."""
    rooms = [r for r in graph.rooms if cat in room_categories(r)]
    orphans = [t for t in graph.texts if t.kind == TextKind.room_label and t.room_id is None
               and cat in label_categories(t.text)]
    return rooms, orphans


def _count_rooms(graph: PlanGraph, intent: str) -> Answer:
    cat = {"saunas": "sauna", "bathrooms": "bathroom", "wc_rooms": "wc", "kitchens": "kitchen",
           "bedrooms": "bedroom", "balconies": "balcony"}[intent]
    rooms, orphans = _rooms_of(graph, cat)
    if not graph.rooms and not orphans and not any(t.kind == TextKind.room_label for t in graph.texts):
        return _prior(intent, "Weder Räume (Layout) noch Raumlabels (OCR) gefunden.")
    parts = []
    by_label = [r for r in rooms if label_categories(r.label_text)]
    by_model = [r for r in rooms if r not in by_label]
    if cat == "balcony" and by_model and _outdoor_labelled(graph):
        # 'Outdoor' also covers terraces, yards and paths. If the drawing names its outdoor
        # spaces (PARVEKE, TERASSI, ...), an unlabelled outdoor area is most likely no balcony.
        parts.append(f"{len(by_model)} unbeschriftete Außenfläche(n) nicht gezählt, da der Plan "
                     "Außenflächen beschriftet (Balkon/Terrasse)")
        rooms = by_label
        by_model = []
    n = len(rooms) + len(orphans)
    if by_label:
        parts.append(f"{len(by_label)} Raum/Räume mit Label " + ", ".join(f"„{r.label_text}“" for r in by_label))
    if by_model:
        parts.append(f"{len(by_model)} ohne passendes Label, Klasse laut Layout-Modell "
                     + ", ".join(f"'{r.model_class}'" for r in by_model))
    if orphans:
        parts.append(f"{len(orphans)} Label(s) ohne erkannten Raum: " + ", ".join(f"„{t.text}“" for t in orphans))
    confs = [r.type_conf for r in by_model] + [0.9] * len(by_label) + [0.6] * len(orphans)
    conf = sum(confs) / len(confs) if confs else 0.6
    return Answer(intent, n, confidence=round(conf, 2),
                  evidence=[r.id for r in rooms] + [t.id for t in orphans],
                  explanation=(f"{n} gezählt: " + "; ".join(parts) + ".") if n else
                  f"Kein Raum der Kategorie '{cat}' gefunden (weder per Label noch per Layout-Modell)"
                  + (f"; {parts[0]}." if parts else "."))


def _outdoor_labelled(graph: PlanGraph) -> bool:
    """Does the drawing name any balcony or terrace (room label or free label text)?"""
    names = [r.label_text for r in graph.rooms] + [t.text for t in graph.texts if t.kind == TextKind.room_label]
    return any(label_categories(x) & {"balcony", "terrace"} for x in names if x)


def _count_showers_bathtubs(graph: PlanGraph, has_layout: bool) -> Answer:
    if not has_layout:
        return _prior("showers_bathtubs", "Layout-Modell nicht gelaufen.")
    tubs = [o for o in graph.objects if o.kind == "bathtub"]
    baths, orphans = _rooms_of(graph, "bathroom")
    tub_rooms = {o.room_id for o in tubs}
    showers = [r for r in baths if r.id not in tub_rooms] + orphans
    n = len(tubs) + len(showers)
    return Answer("showers_bathtubs", n, confidence=0.5 if showers else 0.7,
                  evidence=[o.id for o in tubs] + [getattr(x, "id") for x in showers],
                  explanation=f"{len(tubs)} Badewanne(n) als Symbol erkannt; das Modell kennt kein Duschsymbol, "
                              f"daher je Bad ohne Wanne eine Dusche angenommen ({len(showers)}).")
