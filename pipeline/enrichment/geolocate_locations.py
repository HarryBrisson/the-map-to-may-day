from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import requests

from enrichment.llm_extraction import LLMCallError, estimate_cost_usd
from utils.ids import slugify
from utils.s3_storage import JsonStorage


GOOGLE_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PROMPT_TEMPLATE = "haymarket_location_modernization_v1"


def run_geolocation(
    storage: JsonStorage,
    run_id: str,
    llm_provider: str,
    llm_model: str,
    geocoder: str,
    google_api_key: str = "",
    force: bool = False,
) -> dict[str, Any]:
    locations = storage.read_json("enriched/haymarket/locations/latest.json")
    geocoded_locations, summary = geolocate_locations(
        locations=locations,
        storage=storage,
        run_id=run_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        geocoder=geocoder,
        google_api_key=google_api_key,
        force=force,
    )
    storage.write_json("enriched/haymarket/locations/latest.json", geocoded_locations)
    storage.write_json(f"enriched/haymarket/geolocation/{run_id}.json", summary)
    return {"locations": geocoded_locations, "summary": summary}


def geolocate_locations(
    locations: list[dict[str, Any]],
    storage: JsonStorage,
    run_id: str,
    llm_provider: str,
    llm_model: str,
    geocoder: str,
    google_api_key: str = "",
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if geocoder != "google":
        raise ValueError(f"Unsupported geocoder: {geocoder}")

    seen: set[str] = set()
    updated_locations: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for location in locations:
        location_id = location["id"]
        if location_id in seen:
            continue
        seen.add(location_id)

        if has_google_coordinates(location) and not force:
            updated_locations.append(location)
            records.append({"location_id": location_id, "status": "skipped_existing_google_coordinates"})
            continue

        llm_record = modernize_location_with_llm(
            location=location,
            storage=storage,
            run_id=run_id,
            provider=llm_provider,
            model=llm_model,
        )
        google_record = geocode_with_google(
            location=location,
            modernized=llm_record.get("parsed_output"),
            storage=storage,
            run_id=run_id,
            api_key=google_api_key or os.getenv("GOOGLE_MAPS_API_KEY", ""),
        )
        updated = apply_google_result(location, llm_record, google_record)
        updated_locations.append(updated)
        records.append(
            {
                "location_id": location_id,
                "status": "success" if google_record["status"] == "success" else "error",
                "llm_status": llm_record["status"],
                "google_status": google_record["status"],
                "llm_call_id": llm_record["call_id"],
                "google_response_id": google_record["response_id"],
                "cost_usd": llm_record["cost_usd"],
                "error": google_record.get("error") or llm_record.get("error"),
            }
        )

    summary = build_geolocation_summary(run_id, llm_model, geocoder, records)
    return updated_locations, summary


def modernize_location_with_llm(
    location: dict[str, Any],
    storage: JsonStorage,
    run_id: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    call_id = f"{location['id']}_{slugify(model)}_modern_address"
    input_messages = build_modern_address_messages(location)
    try:
        if provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {provider}")
        parsed_output, raw_output, usage = call_openai_location_modernization(model, input_messages)
        status = "success"
        error = None
    except Exception as exc:
        parsed_output = None
        raw_output = exc.raw_output if isinstance(exc, LLMCallError) else None
        usage = exc.usage if isinstance(exc, LLMCallError) else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        status = "error"
        error = str(exc)

    record = {
        "run_id": run_id,
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_template": PROMPT_TEMPLATE,
        "input_messages": input_messages,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "source_urls": [],
        "usage": usage,
        "cost_usd": round(estimate_cost_usd(model, usage), 8),
        "status": status,
        "error": error,
    }
    storage.write_json(f"raw/haymarket/geolocation/{run_id}/llm/{call_id}.json", record)
    return record


def build_modern_address_messages(location: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You convert historical Chicago location records into modern Google Maps geocoding queries. "
                "Do not invent exact coordinates. Return a query/address only when it is supported by the location record."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a modern geocoding query for this Haymarket-trial location. "
                "Preserve uncertainty in confidence and reasoning.\n\n"
                f"LOCATION JSON:\n{json.dumps(location, ensure_ascii=False)}"
            ),
        },
    ]


def call_openai_location_modernization(
    model: str,
    input_messages: list[dict[str, str]],
) -> tuple[dict[str, Any], Any, dict[str, int]]:
    from openai import OpenAI

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["location_id", "modern_query", "modern_address", "reasoning", "confidence"],
        "properties": {
            "location_id": {"type": "string"},
            "modern_query": {"type": ["string", "null"]},
            "modern_address": {"type": ["string", "null"]},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    response = OpenAI().responses.create(
        model=model,
        input=input_messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "haymarket_location_modernization",
                "schema": schema,
                "strict": True,
            }
        },
    )
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
        raise LLMCallError("OpenAI returned location output that could not be parsed as JSON", raw_output, usage) from exc
    return parsed, raw_output, usage


def geocode_with_google(
    location: dict[str, Any],
    modernized: dict[str, Any] | None,
    storage: JsonStorage,
    run_id: str,
    api_key: str,
) -> dict[str, Any]:
    response_id = f"{location['id']}_google_geocode"
    query = (modernized or {}).get("modern_query") or (modernized or {}).get("modern_address")
    if not query:
        record = google_error_record(run_id, response_id, location, query, "LLM did not produce a modern geocoding query")
        storage.write_json(f"raw/haymarket/geolocation/{run_id}/google/{response_id}.json", record)
        return record
    if not api_key:
        record = google_error_record(run_id, response_id, location, query, "GOOGLE_MAPS_API_KEY is required")
        storage.write_json(f"raw/haymarket/geolocation/{run_id}/google/{response_id}.json", record)
        return record

    params = {"address": query, "key": api_key}
    try:
        response = requests.get(GOOGLE_GEOCODING_URL, params=params, timeout=30)
        raw_output = response.json()
        status = "success" if response.ok and raw_output.get("status") == "OK" and raw_output.get("results") else "error"
        error = None if status == "success" else raw_output.get("error_message") or raw_output.get("status") or response.text
    except Exception as exc:
        raw_output = None
        status = "error"
        error = str(exc)

    record = {
        "run_id": run_id,
        "response_id": response_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "google_maps",
        "location_id": location["id"],
        "query": query,
        "raw_output": raw_output,
        "status": status,
        "error": error,
    }
    storage.write_json(f"raw/haymarket/geolocation/{run_id}/google/{response_id}.json", record)
    return record


def google_error_record(
    run_id: str,
    response_id: str,
    location: dict[str, Any],
    query: str | None,
    error: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "response_id": response_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "google_maps",
        "location_id": location["id"],
        "query": query,
        "raw_output": None,
        "status": "error",
        "error": error,
    }


def apply_google_result(
    location: dict[str, Any],
    llm_record: dict[str, Any],
    google_record: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(location)
    modernized = llm_record.get("parsed_output") or {}
    if modernized.get("modern_address"):
        updated["modern_address"] = modernized["modern_address"]

    if google_record["status"] != "success":
        updated["geocoding"] = {
            "status": "error",
            "llm_call_id": llm_record["call_id"],
            "google_response_id": google_record["response_id"],
            "error": google_record.get("error"),
        }
        return updated

    first_result = google_record["raw_output"]["results"][0]
    geometry = first_result.get("geometry", {})
    location_point = geometry.get("location", {})
    updated["modern_address"] = first_result.get("formatted_address") or updated.get("modern_address")
    updated["coordinates"] = {
        "lat": location_point.get("lat"),
        "lng": location_point.get("lng"),
        "confidence": google_confidence(geometry.get("location_type"), modernized.get("confidence", 0)),
        "method": "google_maps_geocoding",
        "provider": "google_maps",
        "place_id": first_result.get("place_id"),
        "raw_response_id": google_record["response_id"],
    }
    updated["geocoding"] = {
        "status": "success",
        "query": google_record["query"],
        "formatted_address": first_result.get("formatted_address"),
        "location_type": geometry.get("location_type"),
        "llm_call_id": llm_record["call_id"],
        "google_response_id": google_record["response_id"],
    }
    return updated


def google_confidence(location_type: str | None, llm_confidence: float) -> float:
    base = {
        "ROOFTOP": 0.95,
        "RANGE_INTERPOLATED": 0.8,
        "GEOMETRIC_CENTER": 0.7,
        "APPROXIMATE": 0.55,
    }.get(location_type or "", 0.5)
    return round(min(base, float(llm_confidence or base)), 2)


def has_google_coordinates(location: dict[str, Any]) -> bool:
    coordinates = location.get("coordinates") or {}
    return (
        coordinates.get("lat") is not None
        and coordinates.get("lng") is not None
        and coordinates.get("method") == "google_maps_geocoding"
    )


def build_geolocation_summary(
    run_id: str,
    llm_model: str,
    geocoder: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": llm_model,
        "geocoder": geocoder,
        "locations": len(records),
        "successes": sum(1 for record in records if record["status"] == "success"),
        "errors": sum(1 for record in records if record["status"] == "error"),
        "skipped": sum(1 for record in records if record["status"].startswith("skipped")),
        "cost_usd": round(sum(record.get("cost_usd", 0) for record in records), 8),
        "records": records,
    }
