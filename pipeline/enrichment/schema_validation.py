import json
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - dependency may be absent before setup
    Draft202012Validator = None


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def validate_items(items: list[dict[str, Any]], schema_name: str) -> list[str]:
    if Draft202012Validator is None:
        return []

    validator = Draft202012Validator(load_schema(schema_name))
    errors: list[str] = []
    for item in items:
        for error in validator.iter_errors(item):
            item_id = item.get("id", "<missing-id>")
            path = ".".join(str(part) for part in error.path)
            errors.append(f"{schema_name}:{item_id}:{path}:{error.message}")
    return errors
