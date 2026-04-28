from __future__ import annotations

import copy
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from enrichment.progress import StageProgress
from sources.hadc_source import XML_NS, parse_tei_xml, tei_tag
from utils.ids import slugify
from utils.openai_schema import (
    LLMCallError,
    call_openai_structured,
    estimate_cost_usd,
    load_schema_with_refs,
)
from utils.s3_storage import JsonStorage


TAGGING_PROMPT_TEMPLATE = "haymarket_segment_tagging_v1"
TAGGING_MAX_OUTPUT_TOKENS = 1_500
PRIOR_UNIT_CONTEXT = 3
DEFAULT_MAX_WORKERS = 8
DEFAULT_UNIT_ATTEMPTS = 3


@dataclass
class TaggingUnit:
    unit_id: str
    kind: str
    speaker_id: str | None
    page_ref: str | None
    text: str
    tei_snippet: str


@dataclass
class _StageState:
    successes: int = 0
    errors: int = 0
    retried: int = 0
    people: int = 0
    locations: int = 0
    claims: int = 0
    quotes: int = 0
    events: int = 0
    cost_usd: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)


def run_tagging(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    tei_xml: str,
    storage: JsonStorage,
    run_id: str,
    model: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress: StageProgress | None = None,
    max_unit_attempts: int = DEFAULT_UNIT_ATTEMPTS,
) -> dict[str, Any]:
    progress = progress or StageProgress(page["id"], "tagging", enabled=False)
    start = time.monotonic()

    units = split_tei_into_units(tei_xml)
    progress.info(f"{len(units)} units in {max_workers} workers")
    if not units:
        progress.done("0 units", duration_s=0.0)
        return _empty_result(page, tei_xml, model, run_id, briefing)

    schema = load_segment_tags_schema()
    state = _StageState()
    lock = threading.Lock()

    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total = len(units)

    def submit(unit: TaggingUnit, prior_units: list[TaggingUnit]) -> dict[str, Any]:
        return tag_one_unit_with_retry(
            page=page,
            briefing=briefing,
            unit=unit,
            prior_units=prior_units,
            model=model,
            schema=schema,
            max_attempts=max_unit_attempts,
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, unit in enumerate(units):
            prior = units[max(0, index - PRIOR_UNIT_CONTEXT):index]
            future = executor.submit(submit, unit, prior)
            futures[future] = unit

        for future in as_completed(futures):
            unit = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "unit_id": unit.unit_id,
                    "speaker": unit.speaker_id,
                    "status": "error",
                    "error": str(exc),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_usd": 0.0,
                    "tags": None,
                }

            with lock:
                completed += 1
                state.rows.append(row)
                row_usage = row.get("usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                usage_total["input_tokens"] += row_usage.get("input_tokens", 0)
                usage_total["output_tokens"] += row_usage.get("output_tokens", 0)
                usage_total["total_tokens"] += row_usage.get("total_tokens", 0)
                state.cost_usd += float(row.get("cost_usd") or 0.0)
                if int(row.get("attempts") or 1) > 1:
                    state.retried += 1
                if row["status"] == "success":
                    state.successes += 1
                    tags = row.get("tags") or {}
                    state.people += len(tags.get("people", []))
                    state.locations += len(tags.get("locations", []))
                    state.claims += len(tags.get("claims", []))
                    state.events += len(tags.get("events", []))
                    state.quotes += len(tags.get("quotes", []))
                else:
                    state.errors += 1

                if completed % 10 == 0 or completed == total:
                    progress.tick(
                        completed,
                        total,
                        f"people: {state.people}, locations: {state.locations}, claims: {state.claims}",
                    )

    bundle = aggregate_bundle(page, tei_xml, state.rows)

    audit_path = f"raw/haymarket/llm/{run_id}/{slugify(model)}/{page['id']}/tagging.jsonl"
    write_jsonl(storage, audit_path, [_audit_row(row, run_id, page, model) for row in sorted_rows(state.rows, units)])

    duration = time.monotonic() - start
    cost_rounded = round(state.cost_usd, 8)
    retry_suffix = f", retried: {state.retried}" if state.retried else ""
    progress.done(
        f"people: {state.people}, locations: {state.locations}, claims: {state.claims}, "
        f"errors: {state.errors}{retry_suffix}, ${cost_rounded:.4f}",
        duration_s=duration,
    )

    return {
        "bundle": bundle,
        "audit_records": state.rows,
        "audit_path": audit_path,
        "usage": usage_total,
        "cost_usd": cost_rounded,
        "unit_count": total,
        "success_count": state.successes,
        "error_count": state.errors,
        "retried_count": state.retried,
        "duration_s": round(duration, 3),
    }


def tag_one_unit_with_retry(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    unit: TaggingUnit,
    prior_units: list[TaggingUnit],
    model: str,
    schema: dict[str, Any],
    max_attempts: int = DEFAULT_UNIT_ATTEMPTS,
) -> dict[str, Any]:
    """Run tag_one_unit, retrying up to max_attempts on failure.

    Failed attempts still count toward usage and cost (the API call happened),
    so we accumulate usage across all attempts and report the final attempt's
    outcome. attempts and last_error appear in the audit row.
    """
    accumulated_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    accumulated_cost = 0.0
    last_row: dict[str, Any] | None = None
    attempt_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        row = tag_one_unit(
            page=page,
            briefing=briefing,
            unit=unit,
            prior_units=prior_units,
            model=model,
            schema=schema,
        )
        usage = row.get("usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        accumulated_usage["input_tokens"] += usage.get("input_tokens", 0)
        accumulated_usage["output_tokens"] += usage.get("output_tokens", 0)
        accumulated_usage["total_tokens"] += usage.get("total_tokens", 0)
        accumulated_cost += float(row.get("cost_usd") or 0.0)
        last_row = row
        if row["status"] == "success":
            row["usage"] = accumulated_usage
            row["cost_usd"] = round(accumulated_cost, 8)
            row["attempts"] = attempt
            if attempt_errors:
                row["prior_errors"] = attempt_errors
            return row
        attempt_errors.append(row.get("error") or "unknown")

    # All attempts failed; return last row with retry metadata
    final = dict(last_row or {})
    final["usage"] = accumulated_usage
    final["cost_usd"] = round(accumulated_cost, 8)
    final["attempts"] = max_attempts
    final["prior_errors"] = attempt_errors[:-1]  # last error is in row['error'] already
    return final


def tag_one_unit(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    unit: TaggingUnit,
    prior_units: list[TaggingUnit],
    model: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    messages = build_tagging_messages(page, briefing, unit, prior_units)
    try:
        parsed, _raw, usage = call_openai_structured(
            model=model,
            input_messages=messages,
            schema=schema,
            schema_name="haymarket_segment_tags",
            max_output_tokens=TAGGING_MAX_OUTPUT_TOKENS,
        )
        cost_usd = estimate_cost_usd(model, usage)
        return {
            "unit_id": unit.unit_id,
            "speaker": unit.speaker_id,
            "page_ref": unit.page_ref,
            "kind": unit.kind,
            "status": "success",
            "usage": usage,
            "cost_usd": round(cost_usd, 8),
            "tags": parsed,
            "error": None,
        }
    except Exception as exc:
        usage = exc.usage if isinstance(exc, LLMCallError) else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        cost_usd = estimate_cost_usd(model, usage)
        return {
            "unit_id": unit.unit_id,
            "speaker": unit.speaker_id,
            "page_ref": unit.page_ref,
            "kind": unit.kind,
            "status": "error",
            "usage": usage,
            "cost_usd": round(cost_usd, 8),
            "tags": None,
            "error": str(exc),
        }


def build_tagging_messages(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    unit: TaggingUnit,
    prior_units: list[TaggingUnit],
) -> list[dict[str, str]]:
    prior_context = [
        {
            "unit_id": prior.unit_id,
            "speaker": prior.speaker_id,
            "text": prior.text[:600],
        }
        for prior in prior_units
    ]
    return [
        {
            "role": "system",
            "content": (
                "You extract entities, claims, and quotes from one segment of Haymarket trial "
                "testimony. You are given a page-level briefing for global context and the immediately "
                "preceding segments for local context. Only return information directly supported by "
                "the CURRENT segment text — never invent details from the briefing alone. IDs you mint "
                "for new entities/claims/quotes must follow the schema patterns (e.g. 'person_…', "
                "'location_…', 'claim_…'). Reuse IDs from the briefing's speaker_directory when "
                "speakers match. Set source_id on claims/quotes to the page's source_id."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Page source_id: {page['id']}\n"
                f"Current unit_id: {unit.unit_id}\n"
                f"Current speaker_id: {unit.speaker_id or '(unknown)'}\n"
                f"Current page_ref: {unit.page_ref or '(unknown)'}\n\n"
                f"BRIEFING:\n{json.dumps(briefing or {}, ensure_ascii=False)}\n\n"
                f"PRIOR UNITS (most recent last):\n{json.dumps(prior_context, ensure_ascii=False)}\n\n"
                f"CURRENT UNIT TEXT:\n{unit.text}\n\n"
                "Return a JSON object matching the segment_tags schema with: unit_id (echo the "
                "current unit_id), people, locations, claims, events, quotes. Only include entities "
                "actually mentioned in the current unit text. Empty arrays are fine."
            ),
        },
    ]


def split_tei_into_units(tei_xml: str) -> list[TaggingUnit]:
    root = parse_tei_xml(tei_xml)
    body = root.find(f".//{tei_tag('body')}")
    if body is None:
        return []

    units: list[TaggingUnit] = []
    sp_elements = body.findall(f".//{tei_tag('sp')}")
    if sp_elements:
        page_ref_by_position = _build_page_ref_index(body)
        for index, sp in enumerate(sp_elements):
            unit_id = _resolve_unit_id(sp, "sp", index)
            speaker_id = _strip_ref(sp.attrib.get("who"))
            text = _element_text(sp)
            if not text.strip():
                continue
            page_ref = page_ref_by_position.get(id(sp))
            units.append(
                TaggingUnit(
                    unit_id=unit_id,
                    kind="sp",
                    speaker_id=speaker_id,
                    page_ref=page_ref,
                    text=text,
                    tei_snippet=_serialize_fragment(sp),
                )
            )
        return units

    page_ref_by_position = _build_page_ref_index(body)
    p_elements = body.findall(f".//{tei_tag('p')}")
    for index, paragraph in enumerate(p_elements):
        text = _element_text(paragraph)
        if not text.strip():
            continue
        unit_id = _resolve_unit_id(paragraph, "p", index)
        units.append(
            TaggingUnit(
                unit_id=unit_id,
                kind="p",
                speaker_id=None,
                page_ref=page_ref_by_position.get(id(paragraph)),
                text=text,
                tei_snippet=_serialize_fragment(paragraph),
            )
        )
    return units


def aggregate_bundle(page: dict[str, Any], tei_xml: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    people: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    quotes: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        tags = row.get("tags") or {}
        people.extend(tags.get("people", []))
        locations.extend(tags.get("locations", []))
        claims.extend(tags.get("claims", []))
        events.extend(tags.get("events", []))
        quotes.extend(tags.get("quotes", []))

    return {
        "source_id": page["id"],
        "tei_xml": tei_xml,
        "people": _merge_by_id(people),
        "locations": _merge_by_id(locations),
        "claims": _merge_by_id(claims),
        "event_suggestions": _merge_by_id(events),
        "quotes": _merge_by_id(quotes),
    }


def _empty_result(page: dict[str, Any], tei_xml: str, model: str, run_id: str, briefing: dict[str, Any] | None) -> dict[str, Any]:
    del model, run_id, briefing
    return {
        "bundle": {
            "source_id": page["id"],
            "tei_xml": tei_xml,
            "people": [],
            "locations": [],
            "claims": [],
            "event_suggestions": [],
            "quotes": [],
        },
        "audit_records": [],
        "audit_path": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
        "unit_count": 0,
        "success_count": 0,
        "error_count": 0,
        "duration_s": 0.0,
    }


def _merge_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    for item in items:
        item_id = item.get("id")
        if not item_id:
            extras.append(item)
            continue
        if item_id not in merged:
            merged[item_id] = copy.deepcopy(item)
            continue
        merged[item_id] = _merge_item(merged[item_id], item)
    return sorted(merged.values(), key=lambda item: item.get("id", "")) + extras


def _merge_item(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(existing)
    for key, value in incoming.items():
        if value in (None, "", []):
            continue
        current = result.get(key)
        if isinstance(current, list) and isinstance(value, list):
            result[key] = _dedupe_list(current + value)
        elif isinstance(current, dict) and isinstance(value, dict):
            result[key] = _merge_item(current, value)
        elif key == "confidence" and isinstance(value, (int, float)):
            result[key] = max(float(current or 0), float(value))
        elif current in (None, "", []):
            result[key] = value
    return result


def _dedupe_list(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _build_page_ref_index(body: ET.Element) -> dict[int, str | None]:
    index: dict[int, str | None] = {}
    current_ref: str | None = None
    for element in body.iter():
        local = element.tag.split("}", 1)[-1]
        if local == "pb":
            current_ref = element.attrib.get("n") or current_ref
        else:
            index[id(element)] = current_ref
    return index


def _resolve_unit_id(element: ET.Element, prefix: str, index: int) -> str:
    xml_id = element.attrib.get(f"{{{XML_NS}}}id") or element.attrib.get("id")
    if xml_id:
        return xml_id
    n_attr = element.attrib.get("n")
    if n_attr:
        return f"{prefix}_{n_attr}"
    return f"{prefix}_{index:03d}"


def _strip_ref(value: str | None) -> str | None:
    if not value:
        return None
    return value.lstrip("#")


def _element_text(element: ET.Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _serialize_fragment(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode")


def load_segment_tags_schema() -> dict[str, Any]:
    return load_schema_with_refs(
        "segment_tags.schema.json",
        refs={
            "people.items": "person.schema.json",
            "locations.items": "location.schema.json",
            "claims.items": "claim.schema.json",
            "events.items": "event.schema.json",
        },
    )


def write_jsonl(storage: JsonStorage, relative_path: str, rows: list[dict[str, Any]]) -> None:
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    storage.write_text(relative_path, body, "application/x-ndjson; charset=utf-8")


def sorted_rows(rows: list[dict[str, Any]], units: list[TaggingUnit]) -> list[dict[str, Any]]:
    order = {unit.unit_id: index for index, unit in enumerate(units)}
    return sorted(rows, key=lambda row: order.get(row.get("unit_id"), len(order)))


def _audit_row(row: dict[str, Any], run_id: str, page: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "page_id": page["id"],
        "model": model,
        "stage": "tagging",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **row,
    }
