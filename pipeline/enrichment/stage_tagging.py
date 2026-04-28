from __future__ import annotations

import json
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from enrichment.progress import StageProgress
from sources.hadc_source import XML_NS, serialize_xml, tei_tag
from utils.ids import slugify
from utils.openai_schema import (
    LLMCallError,
    call_openai_structured,
    estimate_cost_usd,
    load_segment_tags_schema,
)
from utils.s3_storage import JsonStorage


TAGGING_PROMPT_TEMPLATE = "haymarket_segment_tagging_v1"
TAGGING_MAX_OUTPUT_TOKENS = 1_500
DEFAULT_MAX_WORKERS = 8


@dataclass
class TaggingUnit:
    unit_id: str
    kind: str  # "sp" or "p"
    speaker_id: str | None
    page_ref: str | None
    text: str
    tei_snippet: str


def run_tagging(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    tei_xml: str,
    storage: JsonStorage,
    run_id: str,
    model: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress: StageProgress | None = None,
) -> dict[str, Any]:
    """Run Stage C: per-<sp> tagging with concurrency. Returns:
    {bundle, audit_records, usage, cost_usd, unit_count, status}.
    """
    started = time.perf_counter()
    units = split_tei_into_units(tei_xml)

    if progress:
        progress.info(f"{len(units)} units in {max_workers} workers")

    if not units:
        bundle = _empty_bundle(page, tei_xml)
        return {
            "bundle": bundle,
            "audit_records": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
            "unit_count": 0,
            "status": "success",
        }

    schema = load_segment_tags_schema()
    audit_path = (
        f"raw/haymarket/llm/{run_id}/{slugify(model)}/{page['id']}/tagging.jsonl"
    )
    audit_lock = threading.Lock()
    counts_lock = threading.Lock()
    audit_records: list[dict[str, Any]] = []
    aggregated = {"people": [], "locations": [], "claims": [], "events": [], "quotes": []}
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    completed = 0

    def write_audit_row(row: dict[str, Any]) -> None:
        with audit_lock:
            audit_records.append(row)
            existing = ""
            if storage.exists(audit_path):
                existing = storage.read_text(audit_path)
            storage.write_text(
                audit_path,
                existing + json.dumps(row, ensure_ascii=False) + "\n",
                "application/jsonl; charset=utf-8",
            )

    units_by_id = {unit.unit_id: unit for unit in units}

    def submit(unit: TaggingUnit) -> dict[str, Any]:
        prior = _prior_units(units, unit.unit_id, count=3)
        return tag_one_unit(
            unit=unit,
            prior_units=prior,
            briefing=briefing,
            page=page,
            model=model,
            schema=schema,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(submit, unit): unit for unit in units}
        for future in as_completed(future_map):
            unit = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "unit_id": unit.unit_id,
                    "speaker": unit.speaker_id,
                    "status": "error",
                    "error": str(exc),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_usd": 0.0,
                    "tags": None,
                }

            with counts_lock:
                completed += 1
                for key in ("people", "locations", "claims", "events", "quotes"):
                    items = (result.get("tags") or {}).get(key, []) if result.get("tags") else []
                    aggregated[key].extend(items)
                for key in usage_total:
                    usage_total[key] += result["usage"].get(key, 0)
                cur = completed
                people_n = len(aggregated["people"])
                loc_n = len(aggregated["locations"])
                claim_n = len(aggregated["claims"])

            write_audit_row(result)

            if progress and (cur == 1 or cur % 10 == 0 or cur == len(units)):
                progress.tick(
                    cur,
                    len(units),
                    f"(people: {people_n}, locations: {loc_n}, claims: {claim_n})",
                )

    cost_usd = round(estimate_cost_usd(model, usage_total), 8)
    duration = time.perf_counter() - started

    bundle = {
        "source_id": page["id"],
        "tei_xml": tei_xml,
        "people": _dedupe_by_id(aggregated["people"]),
        "locations": _dedupe_by_id(aggregated["locations"]),
        "claims": _dedupe_by_id(aggregated["claims"]),
        "event_suggestions": _dedupe_by_id(aggregated["events"]),
        "quotes": _materialize_quotes(page, aggregated["quotes"], units_by_id),
    }

    if progress:
        progress.info(
            f"{len(units)}/{len(units)} done in {duration:.1f}s "
            f"(people: {len(bundle['people'])}, "
            f"locations: {len(bundle['locations'])}, "
            f"claims: {len(bundle['claims'])}, ${cost_usd:.4f})"
        )

    return {
        "bundle": bundle,
        "audit_records": audit_records,
        "usage": usage_total,
        "cost_usd": cost_usd,
        "unit_count": len(units),
        "status": "success",
        "duration_s": round(duration, 3),
    }


def tag_one_unit(
    unit: TaggingUnit,
    prior_units: list[TaggingUnit],
    briefing: dict[str, Any] | None,
    page: dict[str, Any],
    model: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    messages = build_tagging_messages(unit, prior_units, briefing, page)
    try:
        parsed, _raw, usage = call_openai_structured(
            model=model,
            input_messages=messages,
            schema=schema,
            schema_name="haymarket_segment_tags",
            max_output_tokens=TAGGING_MAX_OUTPUT_TOKENS,
        )
    except LLMCallError as exc:
        return {
            "unit_id": unit.unit_id,
            "speaker": unit.speaker_id,
            "status": "error",
            "error": str(exc),
            "usage": exc.usage,
            "cost_usd": round(estimate_cost_usd(model, exc.usage), 8),
            "tags": None,
        }
    except Exception as exc:
        return {
            "unit_id": unit.unit_id,
            "speaker": unit.speaker_id,
            "status": "error",
            "error": str(exc),
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
            "tags": None,
        }

    parsed.setdefault("unit_id", unit.unit_id)
    return {
        "unit_id": unit.unit_id,
        "speaker": unit.speaker_id,
        "page_ref": unit.page_ref,
        "status": "success",
        "usage": usage,
        "cost_usd": round(estimate_cost_usd(model, usage), 8),
        "tags": parsed,
    }


def build_tagging_messages(
    unit: TaggingUnit,
    prior_units: list[TaggingUnit],
    briefing: dict[str, Any] | None,
    page: dict[str, Any],
) -> list[dict[str, str]]:
    briefing_payload = briefing or {}
    speaker_directory = briefing_payload.get("speaker_directory") or []
    prior_payload = [
        {"unit_id": u.unit_id, "speaker": u.speaker_id, "text": u.text[:600]}
        for u in prior_units
    ]
    return [
        {
            "role": "system",
            "content": (
                "You extract entities, claims, and quotes from one segment of Haymarket trial "
                "testimony given a document briefing and the immediately preceding context. "
                "Only return information present in the CURRENT segment. Reuse canonical IDs "
                "from the briefing's speaker directory when assigning speaker_person_id; for "
                "people not in the directory, use a deterministic 'person_<lowercase_snake_name>' "
                "ID. Likewise for locations use 'location_<snake_name>'. Return a JSON object "
                "matching the segment_tags schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"SOURCE ID: {page['id']}\n"
                f"BRIEFING JSON:\n{json.dumps(briefing_payload, ensure_ascii=False)}\n\n"
                f"SPEAKER DIRECTORY:\n{json.dumps(speaker_directory, ensure_ascii=False)}\n\n"
                f"PRIOR UNITS (last {len(prior_payload)}):\n"
                f"{json.dumps(prior_payload, ensure_ascii=False)}\n\n"
                f"CURRENT UNIT:\n"
                f"unit_id: {unit.unit_id}\n"
                f"speaker_id: {unit.speaker_id or ''}\n"
                f"page_ref: {unit.page_ref or ''}\n"
                f"text:\n{unit.text}"
            ),
        },
    ]


def split_tei_into_units(tei_xml: str) -> list[TaggingUnit]:
    """Walk the parsed TEI and emit one TaggingUnit per <sp> (or per <p> for
    non-testimony pages with no speech turns)."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    body = root.find(f".//{tei_tag('body')}")
    if body is None:
        return []

    sp_nodes = body.findall(f".//{tei_tag('sp')}")
    units: list[TaggingUnit] = []

    if sp_nodes:
        page_ref: str | None = None
        sp_index = 0
        # Walk the body in document order so we can carry the running page_ref from <pb>.
        for element in body.iter():
            if element.tag == tei_tag("pb"):
                page_ref = element.attrib.get("n") or page_ref
            elif element.tag == tei_tag("sp"):
                sp_index += 1
                unit_id = (
                    element.attrib.get(f"{{{XML_NS}}}id")
                    or element.attrib.get("id")
                    or f"sp_{sp_index:03d}"
                )
                speaker_id = _strip_ref(element.attrib.get("who"))
                text = _element_plain_text(element)
                snippet = serialize_xml(element)
                units.append(
                    TaggingUnit(
                        unit_id=unit_id,
                        kind="sp",
                        speaker_id=speaker_id,
                        page_ref=page_ref,
                        text=text,
                        tei_snippet=snippet,
                    )
                )
        return units

    # Fallback: one unit per <p> inside the body / div.
    page_ref = None
    p_index = 0
    for element in body.iter():
        if element.tag == tei_tag("pb"):
            page_ref = element.attrib.get("n") or page_ref
        elif element.tag == tei_tag("p"):
            p_index += 1
            text = _element_plain_text(element)
            if not text.strip():
                continue
            unit_id = (
                element.attrib.get(f"{{{XML_NS}}}id")
                or element.attrib.get("id")
                or f"p_{p_index:03d}"
            )
            units.append(
                TaggingUnit(
                    unit_id=unit_id,
                    kind="p",
                    speaker_id=None,
                    page_ref=page_ref,
                    text=text,
                    tei_snippet=serialize_xml(element),
                )
            )
    return units


def _element_plain_text(element: ET.Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_element_plain_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(part.strip() for part in parts if part and part.strip())


def _strip_ref(value: str | None) -> str | None:
    if not value:
        return None
    return value.lstrip("#") or None


def _prior_units(units: list[TaggingUnit], unit_id: str, count: int = 3) -> list[TaggingUnit]:
    for index, unit in enumerate(units):
        if unit.unit_id == unit_id:
            start = max(0, index - count)
            return units[start:index]
    return []


def _dedupe_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not item_id:
            continue
        if item_id not in seen:
            seen[item_id] = item
    return list(seen.values())


def _materialize_quotes(
    page: dict[str, Any],
    raw_quotes: list[dict[str, Any]],
    units_by_id: dict[str, TaggingUnit],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, quote in enumerate(raw_quotes):
        if not isinstance(quote, dict):
            continue
        text = (quote.get("quote") or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": f"quote_{page['id']}_{index:04d}",
                "source_id": page["id"],
                "speaker_person_id": quote.get("speaker_person_id"),
                "speaker_label": quote.get("speaker_label"),
                "quote": text,
                "page_refs": quote.get("page_refs", []),
                "confidence": quote.get("confidence", 0.7),
            }
        )
    return out


def _empty_bundle(page: dict[str, Any], tei_xml: str) -> dict[str, Any]:
    return {
        "source_id": page["id"],
        "tei_xml": tei_xml,
        "people": [],
        "locations": [],
        "claims": [],
        "event_suggestions": [],
        "quotes": [],
    }


def build_tagging_audit_record(
    audit_records: list[dict[str, Any]],
    page_id: str,
    model: str,
    run_id: str,
) -> dict[str, Any]:
    """Aggregate one page-level audit summary across all per-unit tagging calls."""
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    successes = 0
    errors = 0
    for record in audit_records:
        for key in total_usage:
            total_usage[key] += record["usage"].get(key, 0)
        total_cost += record.get("cost_usd", 0.0)
        if record["status"] == "success":
            successes += 1
        else:
            errors += 1
    return {
        "run_id": run_id,
        "stage": "tagging",
        "page_id": page_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": TAGGING_PROMPT_TEMPLATE,
        "unit_count": len(audit_records),
        "successes": successes,
        "errors": errors,
        "usage": total_usage,
        "cost_usd": round(total_cost, 8),
        "status": "success" if successes else "error",
        "error": None if successes else "All tagging units failed",
    }
