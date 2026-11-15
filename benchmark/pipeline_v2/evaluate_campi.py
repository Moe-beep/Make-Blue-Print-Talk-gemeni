"""Score schema_campi facts (main.py output) against the dataset questions, with the README's rules.

Every question is mapped to one field of schema_campi.json (in English, German, French, Italian or
Finnish); the value of that field in <facts>/<plan stem>/<file> is the answer. Scoring as in the dataset
README: counts must be exact, areas within 2 %. A question whose field is null counts as unanswered,
one that maps to no field as not covered.

    python evaluate_campi.py data/junction-dataset-preview/questions.jsonl facts
    python evaluate_campi.py questions.jsonl facts --file layout.campi.json     # one model instead of merged
    python evaluate_campi.py questions.jsonl facts --plans held_out.txt         # only the plans listed (stems)
    python evaluate_campi.py questions.jsonl facts --answers answers.jsonl      # also write every answer
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

AREA_TOLERANCE = 0.02  # README: areas within 2 %
MERGED = "merged.campi.json"
ROOM_NUMBER_RE = re.compile(r"\b(U?\d+\.\d{2})\b")
APARTMENT_RE = re.compile(
    r"\b(?:apartment|wohnung|appartement|appartamento)\s+([A-Z0-9]{1,3})\b",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r"[\'\"‘’“”„«»]\s*(.+?)\s*[\'\"‘’“”„«»]")

# (field, keywords): the first entry with a keyword in the folded question wins; order matters
COUNT_FIELDS = [
    # bathrooms first: the English bathroom question says "not counting separate WC rooms"
    (
        "room_type_counts.bathroom_count",
        ("bathrooms or washrooms", "kylpy- tai pesuhuonetta"),
    ),
    ("room_type_counts.wc_room_count", ("separate wc", "wc-tilaa")),
    (
        "sanitary_fixtures.shower_and_bathtub_count",
        ("showers and bathtubs", "suihkua ja kylpyammetta"),
    ),
    (
        "sanitary_fixtures.sink_and_washbasin_count",
        ("sinks and wash", "pesu- ja tiskiallasta"),
    ),
    ("sanitary_fixtures.washbasin_count", ("washbasins", "lavabos", "lavabi")),
    (
        "sanitary_fixtures.toilet_count",
        ("toilet", "wc-istuinta", "wcs", "de wc", "quanti wc"),
    ),
    ("sanitary_fixtures.shower_count", ("showers", "duschen", "douches", "docce")),
    (
        "sanitary_fixtures.bathtub_count",
        ("bathtubs", "badewannen", "baignoires", "vasche da bagno"),
    ),
    ("apartment_count", ("apartments", "wohnungen", "appartements", "appartamenti")),
    ("openings.door_count", ("doors", "turen", "portes", "porte ", "ovea")),
    (
        "openings.window_count",
        ("windows", "fenster", "fenetres", "finestre", "ikkunaa"),
    ),
    ("floor_count", ("floors (storeys)", "kerrosta", "etage")),
    ("room_type_counts.sauna_count", ("saunas", "saunaa")),
    ("room_type_counts.kitchen_count", ("kitchens", "keittiota")),
    ("room_type_counts.bedroom_count", ("bedrooms", "makuuhuonetta")),
    ("room_type_counts.balcony_count", ("balconies", "parveketta")),
]
LABEL_WORDS = (
    "labelled",
    "bezeichnung",
    "designation",
    "denominazione",
    "denominati",
    "designees",
)
TOTAL_WORDS = ("numbered indoor", "nummerierten innen", "numerotees", "numerati")


def fold(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    )


def get(doc, dotted: str):
    for part in dotted.split("."):
        doc = doc.get(part) if isinstance(doc, dict) else None
    return doc


def entry_value(fp: dict, table: str, key: str, field: str):
    return ((fp.get(table) or {}).get(key) or {}).get(field)


def plan_label(question: str, by_label: dict) -> str | None:
    """Longest room label of the plan written in the question (labels may contain apostrophes)."""
    found = [label for label in by_label if label in question]
    if found:
        return max(found, key=len)
    match = QUOTE_RE.search(question)
    return match.group(1) if match else None


def predict(q: dict, fp: dict) -> tuple[str, object, str | None]:
    """(kind of question, value from the facts, schema field); kind 'not_covered' when no field answers it."""
    text, folded = q["question"], fold(q["question"])
    by_label = fp.get("rooms_by_label") or {}
    if q["type"] == "area":
        if any(w in folded for w in TOTAL_WORDS):
            return (
                "area_floor_total",
                fp.get("total_net_floor_area_m2"),
                "floor_plan.total_net_floor_area_m2",
            )
        apartment = APARTMENT_RE.search(text)
        if apartment:
            # keys contain dots (room 5.11): look them up directly, not as a dotted path
            return (
                "area_apartment",
                entry_value(fp, "apartments", apartment.group(1), "net_floor_area_m2"),
                "floor_plan.apartments",
            )
        room = ROOM_NUMBER_RE.search(text)
        if room:
            return (
                "area_room",
                entry_value(fp, "rooms", room.group(1), "net_floor_area_m2"),
                "floor_plan.rooms",
            )
        label = plan_label(text, by_label)
        if label is not None:
            entry = by_label.get(label)
            return (
                "area_label",
                entry.get("total_net_area_m2") if entry else None,
                "floor_plan.rooms_by_label",
            )
        return "not_covered", None, None
    if any(w in folded for w in LABEL_WORDS):
        label = plan_label(text, by_label)
        if not by_label:
            return "count_label", None, "floor_plan.rooms_by_label"
        entry = by_label.get(label)
        return (
            "count_label",
            entry.get("room_count") if entry else 0,
            "floor_plan.rooms_by_label",
        )
    for field, words in COUNT_FIELDS:
        if any(w in folded for w in words):
            return field.split(".")[-1], get(fp, field), f"floor_plan.{field}"
    return "not_covered", None, None


def correct(q: dict, value) -> bool:
    answer = float(q["answer"])
    if q["type"] == "count":
        return float(value) == answer
    return abs(float(value) - answer) <= AREA_TOLERANCE * abs(answer)


def answer_record(
    q: dict, value, field: str | None, model: str | None, verdict: str
) -> dict:
    """One line of the answers file: the question, our answer and where it comes from."""
    if value is not None:
        value = int(value) if q["type"] == "count" else round(float(value), 2)
    return {
        "path": q["path"],
        "question": q["question"],
        "type": q["type"],
        "unit": q.get("unit"),
        "answer": value,  # null = we do not know (no drawing, no field, or the field is unknown)
        "field": field,  # schema_campi.json field the answer was read from
        "model": model,  # source of that field in merged.campi.json
        "expected": q["answer"],
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("facts", type=Path, help="main.py --output folder")
    parser.add_argument(
        "--file", default=MERGED, help=f"facts file per plan (default {MERGED})"
    )
    parser.add_argument(
        "--plans", type=Path, help="text file with plan stems to keep (one per line)"
    )
    parser.add_argument(
        "--answers",
        type=Path,
        help="write one answer per question (all of them) to this .jsonl file",
    )
    args = parser.parse_args()
    keep = set(args.plans.read_text().split()) if args.plans else None

    by_kind: dict[tuple[str, str], Counter] = defaultdict(Counter)
    answers: list[dict] = []
    facts_cache: dict[str, dict | None] = {}
    for line in args.questions.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(q["path"]).stem)
        if keep is not None and stem not in keep:
            continue
        if stem not in facts_cache:
            path = args.facts / stem / args.file
            facts_cache[stem] = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            )
        doc = facts_cache[stem]
        if doc is None:
            # no drawing for this question in the facts folder
            answers.append(answer_record(q, None, None, None, "no_drawing"))
            continue
        fp = doc["floor_plan"]
        source = q["path"].split("/")[1]
        kind, value, field = predict(q, fp)
        verdict = (
            "not_covered"
            if kind == "not_covered"
            else "unanswered"
            if value is None
            else "correct"
            if correct(q, value)
            else "wrong"
        )
        by_kind[(source, kind)][verdict] += 1
        model = (
            (doc.get("_meta", {}).get("sources") or {}).get(field) if field else None
        )
        answers.append(answer_record(q, value, field, model, verdict))

    print(
        f"{'source':16}{'question kind':30}{'correct':>9}{'wrong':>7}{'null':>7}{'total':>7}{'score':>8}"
    )
    totals: dict[str, Counter] = defaultdict(Counter)
    for (source, kind), c in sorted(by_kind.items()):
        n = sum(c.values())
        totals[source].update(c)
        print(
            f"{source:16}{kind:30}{c['correct']:>9}{c['wrong']:>7}{c['unanswered']:>7}{n:>7}{c['correct'] / n:>8.1%}"
        )
    print()
    grand = Counter()
    for source, c in totals.items():
        n = sum(c.values())
        grand.update(c)
        print(
            f"{source:16}{'ALL':30}{c['correct']:>9}{c['wrong']:>7}{c['unanswered']:>7}{n:>7}{c['correct'] / n:>8.1%}"
        )
    if args.answers:
        lines = [json.dumps(a, ensure_ascii=False) for a in answers]
        args.answers.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n{len(answers)} answers -> {args.answers}")
    n = sum(grand.values())
    if n:
        print(
            f"{'ALL':16}{'':30}{grand['correct']:>9}{grand['wrong']:>7}{grand['unanswered']:>7}{n:>7}{grand['correct'] / n:>8.1%}"
        )


if __name__ == "__main__":
    main()
