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


def load_schema_with_refs(name: str, refs: dict[str, str] | None = None) -> dict[str, Any]:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    if refs:
        for property_path, ref_name in refs.items():
            ref_schema = json.loads((SCHEMA_DIR / ref_name).read_text(encoding="utf-8"))
            inject_ref(schema, property_path.split("."), ref_schema)
    normalize_openai_schema(schema)
    return schema


def inject_ref(schema: dict[str, Any], path: list[str], ref_schema: dict[str, Any]) -> None:
    cursor: Any = schema
    for part in path[:-1]:
        if part == "items":
            cursor = cursor["items"]
        else:
            cursor = cursor["properties"][part]
    last = path[-1]
    if last == "items":
        cursor["items"] = ref_schema
    else:
        cursor["properties"][last] = ref_schema


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


def call_openai_text(
    model: str,
    input_messages: list[dict[str, str]],
    max_output_tokens: int | None = None,
) -> tuple[str, Any, dict[str, int]]:
    """Plain-text completion. Use when the response is not naturally JSON
    (e.g. raw TEI XML) — avoids the JSON-string-escape token tax that
    structured outputs impose."""
    from openai import OpenAI

    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "input": input_messages,
    }
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**request)
    raw_output = response.model_dump(mode="json")
    usage_obj = raw_output.get("usage") or {}
    input_tokens = usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens") or 0
    output_tokens = usage_obj.get("output_tokens") or usage_obj.get("completion_tokens") or 0
    usage = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(usage_obj.get("total_tokens") or input_tokens + output_tokens),
    }
    return response.output_text, raw_output, usage


def call_openai_structured(
    model: str,
    input_messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int | None = None,
) -> tuple[dict[str, Any], Any, dict[str, int]]:
    from openai import OpenAI

    client = OpenAI()
    request: dict[str, Any] = {
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
        request["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**request)
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
        raise LLMCallError("OpenAI returned output that could not be parsed as JSON", raw_output, usage) from exc
    return parsed, raw_output, usage


MODEL_PRICING_PER_1M = {
    # GPT-4 family (legacy but still used for cheap tasks)
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    # GPT-5 family
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00},
    # o-series reasoning models
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o3": {"input": 2.00, "output": 8.00},
}


def estimate_cost_usd(model: str, usage: dict[str, int]) -> float:
    pricing = MODEL_PRICING_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (usage["input_tokens"] / 1_000_000 * pricing["input"]) + (usage["output_tokens"] / 1_000_000 * pricing["output"])
