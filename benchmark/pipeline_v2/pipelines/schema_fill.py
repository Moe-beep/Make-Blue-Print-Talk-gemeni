"""schema_campi.json as an empty facts document: every field null, maps and lists empty.

pipelines/campi_fill.py fills it from one model by the rules; pipelines/merge_campi.py combines the models.
"""

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema_campi.json"


def load_schema(path: Path = SCHEMA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["schema"]


def skeleton(node: dict) -> object:
    """Empty value for one schema node: objects expand, maps/lists start empty, leaves are null."""
    node_type = node.get("type")
    if node_type == "object":
        return {key: skeleton(child) for key, child in node.get("properties", {}).items()}
    if node_type == "map":
        return {}
    if node_type == "list":
        return []
    return None


def empty_facts(schema: dict) -> dict:
    return {section: skeleton(node) for section, node in schema.items()}
