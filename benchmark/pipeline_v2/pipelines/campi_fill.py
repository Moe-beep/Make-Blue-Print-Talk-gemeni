"""Fill schema_campi.json from one building-graph/1 document by applying the rules of planqa/campi_map.py.

For every field of the schema the first matching rule decides:
  fact   `where` is a JSON path (alternatives separated by " / ", list positions as numbers, e.g.
         source.image_size.0): the first one present is copied
  query  table[column] (spaces, elements.<group>, storeys):
           list of objects -> one entry per row of that table
           map of objects  -> one entry per row, keyed by the column named in the rule note
                              ("keyed by spaces[number]")
  maps of objects with a fact rule (facts.areas.by_name, facts.areas.by_unit) are copied entry by entry,
         the schema property names mapped to the building-graph names by FIELD_ALIASES
  derive / interpret / gap / external: left null (they need the agent, an extractor that does not
         exist yet, or data outside the drawing)
Only the schema sections that the kind of source can describe are filled (SECTIONS_BY_KIND).
_meta.filled lists every filled field with the building-graph location it came from.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import UTC, datetime

from pipelines.schema_fill import empty_facts
from planqa.campi_map import RULES

SECTIONS_BY_KIND = {
    "image": ("floor_plan",),
    "pdf": ("floor_plan",),
    "ifc": ("floor_plan", "bim_model"),
}
# schema property -> building-graph entry key, for maps copied from facts.areas.*
FIELD_ALIASES = {
    "room_count": "n",
    "total_net_area_m2": "net_area",
    "net_floor_area_m2": "net_area",
}
TABLE_COLUMN_RE = re.compile(r"^([a-z_]+(?:\.[a-z_]+)?)\[(\w+)\]$")
KEY_COLUMN_RE = re.compile(r"keyed by [a-z_.]+\[(\w+)\]")
NUMBER_TYPES = {"int": int, "real": float}
# values shown as written on drawings; sums are computed from unrounded areas before this
UNIT_DECIMALS = {"m2": 2, "m3": 2, "m": 2}


def rule_for(path: str) -> tuple[str, str, str, str] | None:
    return next((r for r in RULES if fnmatch.fnmatchcase(path, r[0])), None)


def resolve(doc: dict, dotted: str):
    node = doc
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def table_rows(doc: dict, table: str) -> list[dict] | None:
    """Rows of spaces / elements.<group> as dicts, storeys as they are; None when the table is absent."""
    if table == "storeys":
        return doc.get("storeys")
    data = resolve(doc, table)
    if not isinstance(data, dict) or "columns" not in data:
        return None
    return [dict(zip(data["columns"], row)) for row in data.get("rows", [])]


def flatten(values) -> list:
    out = []
    for v in values:
        out.extend(flatten(v) if isinstance(v, (list, tuple)) else [v])
    return out


def coerce(value, spec: dict):
    """Value in the schema type of spec, or None when it does not fit (never guess)."""
    kind, unit = spec.get("type"), spec.get("unit")
    if kind == "list":
        if not isinstance(value, (list, tuple)):
            return None
        items = [coerce(v, spec["items"]) for v in flatten(value)]
        return items if items and None not in items else None
    if value is None or isinstance(value, (dict, list)):
        return None
    if kind in NUMBER_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if kind == "int" and float(value) != int(value):
            return None
        number = NUMBER_TYPES[kind](value)
        return round(number, UNIT_DECIMALS[unit]) if unit in UNIT_DECIMALS else number
    if kind == "bool":
        return value if isinstance(value, bool) else None
    return str(value)


def fill_scalar(doc: dict, path: str, spec: dict) -> tuple[object, str] | None:
    rule = rule_for(path)
    if rule is None or rule[1] != "fact":
        return None
    for where in (w.strip() for w in rule[2].split("/")):
        value = coerce(resolve(doc, where), spec)
        if value is not None:
            return value, where
    return None


def query_columns(rules: dict) -> tuple[str, dict[str, str]] | None:
    """(table, {property: column}) when every property is a query on columns of the same table."""
    parsed = {p: TABLE_COLUMN_RE.match(r[2].strip()) for p, r in rules.items()}
    if any(m is None for m in parsed.values()):
        return None
    tables = {m.group(1) for m in parsed.values()}
    return (
        (tables.pop(), {p: m.group(2) for p, m in parsed.items()})
        if len(tables) == 1
        else None
    )


def object_rules(path: str, props: dict, separator: str) -> dict | None:
    rules = {p: rule_for(f"{path}.{separator}.{p}") for p in props}
    return None if not props or any(r is None for r in rules.values()) else rules


def row_item(row: dict, columns: dict[str, str], props: dict) -> dict:
    return {p: coerce(row.get(c), props[p]) for p, c in columns.items()}


def fill_map(doc: dict, path: str, node: dict) -> tuple[dict, str] | None:
    """A map of objects: from a facts dict (fact rules) or from table rows (query rules)."""
    props = node["items"].get("properties", {})
    rules = object_rules(path, props, "*")
    if rules is None:
        return None
    statuses = {r[1] for r in rules.values()}
    if statuses == {"fact"}:
        where = next(iter(rules.values()))[2].strip()
        source = resolve(doc, where)
        if not isinstance(source, dict) or not source:
            return None
        out = {
            key: {
                p: coerce(entry.get(FIELD_ALIASES.get(p, p)), spec)
                for p, spec in props.items()
            }
            for key, entry in source.items()
        }
        return out, where
    if statuses != {"query"}:
        return None
    query = query_columns(rules)
    keys = {
        m.group(1) if (m := KEY_COLUMN_RE.search(r[3])) else None
        for r in rules.values()
    }
    if query is None or len(keys) != 1 or None in keys:
        return None
    table, columns = query
    key_column = keys.pop()
    out = {
        str(row[key_column]): row_item(row, columns, props)
        for row in table_rows(doc, table) or []
        if row.get(key_column) not in (None, "")
    }
    return (out, f"{table} keyed by {key_column}") if out else None


def fill_list(doc: dict, path: str, node: dict) -> tuple[list, str] | None:
    """A list of objects from the rows of one table (query rules); lists of values are filled in coerce."""
    props = node["items"].get("properties", {})
    rules = object_rules(path, props, "[]")
    if rules is None or {r[1] for r in rules.values()} != {"query"}:
        return None
    query = query_columns(rules)
    if query is None:
        return None
    table, columns = query
    rows = table_rows(doc, table)
    if rows is None:
        return None
    items = [row_item(row, columns, props) for row in rows]
    items = [item for item in items if any(v is not None for v in item.values())]
    return (items, table) if items else None


def fill_node(
    doc: dict, node: dict, path: str, target: dict, key: str, filled: dict
) -> None:
    kind = node.get("type")
    if kind == "object":
        for child_key, child in node.get("properties", {}).items():
            fill_node(doc, child, f"{path}.{child_key}", target[key], child_key, filled)
        return
    if kind in ("map", "list") and node["items"].get("type") == "object":
        result = (fill_map if kind == "map" else fill_list)(doc, path, node)
    elif kind in ("map", "list"):
        result = None  # maps/lists of plain values have no mechanical rule yet
    else:
        result = fill_scalar(doc, path, node)
    if result:
        target[key], filled[path] = result


def fill_campi(doc: dict, schema: dict, model: str) -> dict:
    """schema_campi facts for one building-graph document (as JSON dict) of one model."""
    facts = empty_facts(schema)
    kind = doc["source"]["kind"]
    sections = SECTIONS_BY_KIND.get(kind, ())
    filled: dict[str, str] = {}
    for section in sections:
        holder = {section: facts[section]}
        fill_node(doc, schema[section], section, holder, section, filled)
        facts[section] = holder[section]
    meta = {
        "schema": "schema_campi.json",
        "rules": "planqa/campi_map.py",
        "model": model,
        "file": doc["source"]["file"],
        "source_kind": kind,
        "sections": list(sections),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "filled": filled,
        "not_measured_by_model": doc.get("facts", {}).get("not_measured", []),
    }
    return {"_meta": meta, **facts}
