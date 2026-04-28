from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PIPELINE_ROOT / "schemas"

OPENAI_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "title",
    "examples",
    "default",
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
}


class LLMCallError(Exception):
    def __init__(self, message: str, raw_output: Any | None = None, usage: dict[str, int] | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.usage = usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def normalize_openai_schema(schema: Any) -> None:
    if isinstance(schema, dict):
        for key in OPENAI_UNSUPPORTED_SCHEMA_KEYS:
            schema.pop(key, None)

        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = set(schema.get("required", properties.keys()))
            for name in list(properties):
                if name not in required:
                    properties.pop(name)

            schema["additionalProperties"] = False
            schema["required"] = list(properties.keys())

            for property_schema in properties.values():
                normalize_openai_schema(property_schema)

        items = schema.get("items")
        if items is not None:
            normalize_openai_schema(items)

        additional_properties = schema.get("additionalProperties")
        if isinstance(additional_properties, dict):
            normalize_openai_schema(additional_properties)

        for key in ["anyOf", "oneOf", "allOf"]:
            for value in schema.get(key, []):
                normalize_openai_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            normalize_openai_schema(value)


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def load_extraction_bundle_schema() -> dict[str, Any]:
    schema = load_schema("extraction_bundle.schema.json")
    refs = {
        "person.schema.json": load_schema("person.schema.json"),
        "location.schema.json": load_schema("location.schema.json"),
        "claim.schema.json": load_schema("claim.schema.json"),
        "event.schema.json": load_schema("event.schema.json"),
    }
    for collection, ref_name in [
        ("people", "person.schema.json"),
        ("locations", "location.schema.json"),
        ("claims", "claim.schema.json"),
        ("event_suggestions", "event.schema.json"),
    ]:
        schema["properties"][collection]["items"] = refs[ref_name]

    normalize_openai_schema(schema)
    return schema


def load_briefing_schema() -> dict[str, Any]:
    schema = load_schema("briefing.schema.json")
    normalize_openai_schema(schema)
    return schema


def load_segment_tags_schema() -> dict[str, Any]:
    schema = load_schema("segment_tags.schema.json")
    refs = {
        "person.schema.json": load_schema("person.schema.json"),
        "location.schema.json": load_schema("location.schema.json"),
        "claim.schema.json": load_schema("claim.schema.json"),
        "event.schema.json": load_schema("event.schema.json"),
    }
    for collection, ref_name in [
        ("people", "person.schema.json"),
        ("locations", "location.schema.json"),
        ("claims", "claim.schema.json"),
        ("events", "event.schema.json"),
    ]:
        schema["properties"][collection]["items"] = refs[ref_name]
    normalize_openai_schema(schema)
    return schema


def call_openai_structured(
    model: str,
    input_messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int | None = None,
) -> tuple[dict[str, Any], Any, dict[str, int]]:
    from openai import OpenAI

    client = OpenAI()
    kwargs: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**kwargs)
    raw_output = response.model_dump(mode="json")
    usage_obj = raw_output.get("usage") or {}
    input_tokens = usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens") or 0
    output_tokens = usage_obj.get("output_tokens") or usage_obj.get("completion_tokens") or 0
    usage = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(usage_obj.get("total_tokens") or input_tokens + output_tokens),
    }
    try:
        parsed = json.loads(response.output_text)
    except Exception as exc:
        raise LLMCallError(
            "OpenAI returned output that could not be parsed as JSON", raw_output, usage
        ) from exc
    return parsed, raw_output, usage


MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


def estimate_cost_usd(model: str, usage: dict[str, int]) -> float:
    pricing = MODEL_PRICING_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (
        usage["input_tokens"] / 1_000_000 * pricing["input"]
        + usage["output_tokens"] / 1_000_000 * pricing["output"]
    )
