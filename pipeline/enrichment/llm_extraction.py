from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enrichment.tei_validation import TEIValidationError, validate_generated_tei_or_raise
from sources.hadc_source import XML_NS, extract_page_sections, serialize_xml, tei_tag
from utils.ids import slugify
from utils.s3_storage import JsonStorage


PROMPT_TEMPLATE = "haymarket_tei_extraction_v2"
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PIPELINE_ROOT / "schemas"
FULL_DOCUMENT_CHAR_LIMIT = 120_000
RAW_HTML_EXCERPT_CHARS = 8_000

MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}
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


def extract_pages_with_audit(
    pages: list[dict[str, Any]],
    storage: JsonStorage,
    run_id: str,
    provider: str,
    models: list[str],
) -> dict[str, Any]:
    audit_records: list[dict[str, Any]] = []
    bundles_by_model: dict[str, list[dict[str, Any]]] = {}

    for model in models:
        bundles_by_model[model] = []
        for page in pages:
            call = extract_one_page(page=page, storage=storage, run_id=run_id, provider=provider, model=model)
            audit_records.append(call)
            if isinstance(call["parsed_output"], dict):
                bundles_by_model[model].append(call["parsed_output"])

            model_path = slugify(model)
            storage.write_json(f"raw/haymarket/llm/{run_id}/{model_path}/{call['call_id']}.json", call)
            print_llm_call_summary(call)

    cost_summary = build_cost_summary(run_id, audit_records)
    model_eval = build_model_eval(run_id, bundles_by_model, audit_records)
    storage.write_json(f"enriched/haymarket/llm_costs/{run_id}.json", cost_summary)
    storage.write_json(f"enriched/haymarket/model_evals/{run_id}.json", model_eval)

    return {
        "bundles_by_model": bundles_by_model,
        "audit_records": audit_records,
        "cost_summary": cost_summary,
        "model_eval": model_eval,
    }


def print_llm_call_summary(call: dict[str, Any]) -> None:
    diagnostics = call.get("input_diagnostics", {})
    parsed = call.get("parsed_output") or {}
    validation = parsed.get("tei_validation") or ((call.get("raw_output") or {}).get("tei_validation") if isinstance(call.get("raw_output"), dict) else {})
    text_validation = validation.get("text", {}) if isinstance(validation, dict) else {}
    print(
        "LLM TEI "
        f"{call['call_id']}: status={call['status']}, "
        f"chunks={diagnostics.get('chunk_count', 0)} {diagnostics.get('chunk_modes', [])}, "
        f"sent_chars={diagnostics.get('sent_chars', 0)}, "
        f"tei_text_ratio={text_validation.get('similarity_ratio', 'n/a')}, "
        f"people={len(parsed.get('people', []))}, "
        f"locations={len(parsed.get('locations', []))}, "
        f"claims={len(parsed.get('claims', []))}, "
        f"events={len(parsed.get('event_suggestions', []))}, "
        f"quotes={len(parsed.get('quotes', []))}"
    )


def extract_one_page(
    page: dict[str, Any],
    storage: JsonStorage,
    run_id: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    call_id = f"{page['id']}_{slugify(model)}"
    raw_html = storage.read_text(page["raw_html_path"]) if page.get("raw_html_path") else ""
    chunks = build_source_chunks(page, raw_html)
    input_messages = [message for chunk in chunks for message in build_messages(page, chunk)]
    parsed_chunks: list[dict[str, Any]] = []
    raw_outputs: list[Any] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        if provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {provider}")

        for chunk in chunks:
            chunk_messages = build_messages(page, chunk)
            parsed_chunk, raw_output, chunk_usage = call_openai_structured(model, chunk_messages)
            parsed_chunks.append(parsed_chunk)
            raw_outputs.append(raw_output)
            usage["input_tokens"] += chunk_usage["input_tokens"]
            usage["output_tokens"] += chunk_usage["output_tokens"]
            usage["total_tokens"] += chunk_usage["total_tokens"]

        parsed_output = combine_chunk_outputs(page, parsed_chunks)
        validation = validate_generated_tei_or_raise(page, parsed_output["tei_xml"])
        parsed_output["tei_validation"] = validation
        raw_output = raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs
        status = "success"
        error = None

        cost_usd = estimate_cost_usd(model, usage)
    except TEIValidationError as exc:
        parsed_output = None
        raw_output = {"tei_validation": exc.validation}
        usage = usage
        cost_usd = estimate_cost_usd(model, usage)
        status = "error"
        error = str(exc)
    except Exception as exc:
        parsed_output = None
        raw_output = exc.raw_output if isinstance(exc, LLMCallError) else None
        usage = exc.usage if isinstance(exc, LLMCallError) else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        cost_usd = estimate_cost_usd(model, usage)
        status = "error"
        error = str(exc)

    return {
        "run_id": run_id,
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_template": PROMPT_TEMPLATE,
        "input_messages": input_messages,
        "input_diagnostics": build_input_diagnostics(page, chunks, raw_html),
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "source_urls": [page["url"]],
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
        "status": status,
        "error": error,
    }


def build_messages(page: dict[str, Any], chunk: dict[str, Any] | None = None) -> list[dict[str, str]]:
    chunk = chunk or build_source_chunks(page, "")[0]
    source_structure = {
        "transcript_metadata": page.get("transcript_metadata", {}),
        "source_stats": page.get("source_stats", {}),
        "candidate_text_sha256": page.get("candidate_text_sha256"),
        "page_cues": chunk.get("page_cues", []),
        "speaker_context": chunk.get("speaker_context", {}),
        "transcript_artifacts": {
            "tei_path": page.get("tei_path"),
            "transcript_json_path": page.get("transcript_json_path"),
        },
        "toc_entries": page.get("toc_entries", [])[:60],
    }
    return [
        {
            "role": "system",
            "content": (
                "You convert messy Haymarket trial source HTML/text into TEI XML and extract structured "
                "historical evidence. Preserve source wording exactly except for whitespace normalization. "
                "Use TEI body markup for page breaks, speaker turns, and inline evidence spans. "
                "Every answer/quote/claim must include speaker attribution when the source context supports it. "
                "Return only data supported by the source."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source ID: {page['id']}\n"
                f"URL: {page['url']}\n"
                f"Title: {page['title']}\n"
                f"Type: {page['source_type']}\n\n"
                "Return a JSON object with keys: source_id, tei_xml, people, locations, claims, "
                "event_suggestions, quotes.\n"
                "The tei_xml must be well-formed TEI. Include pb elements for page cues, sp/speaker/p for "
                "testimony, sp@who where the speaker is known, inline seg elements for people, locations, "
                "events, and claim evidence, and standOff lists/annotations for provisional entities, claims, "
                "and quotes. Use provisional IDs where canonical IDs are uncertain.\n"
                "Claims should include reporter/speaker person IDs, claim date, event time, location, subject "
                "people, statement, quote, page refs, and confidence. Quotes must include speaker_person_id, "
                "speaker_label, quote text, and page refs when available.\n"
                "Only put underlying historical happenings in event_suggestions, such as meetings, speeches, "
                "police movements, orders to disperse, explosions, shootings, arrests, or courtroom proceedings "
                "when the proceeding itself is the mapped event. Do not turn exhibit metadata, document descriptions, "
                "or the mere fact that testimony was given into map events. Evidence introductions belong in claims.\n"
                "When a location refers to Haymarket between Desplaines and Randolph streets, use a stable location "
                "record with address_1886 set to that historic text. Coordinates may be null if the source does not "
                "support them; if you provide coordinates, explain the basis in coordinates.method.\n\n"
                f"SOURCE STRUCTURE JSON:\n{json.dumps(source_structure, ensure_ascii=False)}\n\n"
                f"RAW HTML EXCERPT:\n{chunk.get('raw_html_excerpt', '')}\n\n"
                f"CANDIDATE SOURCE TEXT ({chunk['mode']}, pages {chunk.get('page_range') or 'all'}):\n"
                f"{chunk['text']}"
            ),
        },
    ]


def build_source_chunks(page: dict[str, Any], raw_html: str, max_chars: int = FULL_DOCUMENT_CHAR_LIMIT) -> list[dict[str, Any]]:
    text = page.get("text") or ""
    raw_html_excerpt = raw_html[:RAW_HTML_EXCERPT_CHARS]
    if len(text) <= max_chars:
        return [
            {
                "index": 0,
                "mode": "full_document",
                "text": text,
                "raw_html_excerpt": raw_html_excerpt,
                "page_cues": page.get("page_cues", []),
                "page_range": page_range(page.get("page_cues", [])),
                "speaker_context": build_speaker_context(page, text),
            }
        ]

    sections = extract_page_sections(text)
    if not sections:
        sections = [{"index": 0, "text": text, "page_ref": None}]

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for section in sections:
        section_text = section.get("text", "")
        if current and current_chars + len(section_text) > max_chars:
            chunks.append(build_chunk(page, chunks, current, raw_html_excerpt, text))
            current = []
            current_chars = 0
        current.append(section)
        current_chars += len(section_text)
    if current:
        chunks.append(build_chunk(page, chunks, current, raw_html_excerpt, text))
    return chunks


def build_chunk(page: dict[str, Any], chunks: list[dict[str, Any]], sections: list[dict[str, Any]], raw_html_excerpt: str, full_text: str) -> dict[str, Any]:
    page_refs = [section.get("page_ref") for section in sections if section.get("page_ref")]
    cues = [cue for cue in page.get("page_cues", []) if not page_refs or cue.get("page_ref") in page_refs]
    text = "\n".join(section.get("text", "") for section in sections if section.get("text"))
    return {
        "index": len(chunks),
        "mode": "page_range_chunk",
        "text": text,
        "raw_html_excerpt": raw_html_excerpt if not chunks else "",
        "page_cues": cues,
        "page_range": page_range(cues),
        "speaker_context": build_speaker_context(page, full_text),
    }


def page_range(page_cues: list[dict[str, Any]]) -> str | None:
    refs = [str(cue.get("page_ref")) for cue in page_cues if cue.get("page_ref")]
    if not refs:
        return None
    return refs[0] if len(refs) == 1 else f"{refs[0]}-{refs[-1]}"


def build_speaker_context(page: dict[str, Any], text: str) -> dict[str, Any]:
    metadata = page.get("transcript_metadata", {})
    witness = metadata.get("witness_name")
    context = {
        "witness_name": witness,
        "question_speaker": "examining attorney",
        "answer_speaker": witness,
        "source_type": page.get("source_type"),
    }
    named_speakers = sorted({match.group(1).strip() for match in re_finditer_speaker(text)})
    context["named_speakers"] = named_speakers[:25]
    return context


def re_finditer_speaker(text: str):
    import re

    return re.finditer(r"^((?:MR|Mr|THE COURT|WITNESS|A JUROR)[^:\n]*):", text, flags=re.MULTILINE)


def build_input_diagnostics(page: dict[str, Any], chunks: list[dict[str, Any]], raw_html: str) -> dict[str, Any]:
    sent_chars = sum(len(chunk["text"]) + len(chunk.get("raw_html_excerpt", "")) for chunk in chunks)
    flagged = page.get("source_stats", {})
    return {
        "raw_html_chars": len(raw_html),
        "candidate_text_chars": len(page.get("text") or ""),
        "sent_chars": sent_chars,
        "chunk_count": len(chunks),
        "chunk_modes": [chunk["mode"] for chunk in chunks],
        "page_ranges": [chunk.get("page_range") for chunk in chunks],
        "page_markers": flagged.get("page_markers", 0),
    }


def combine_chunk_outputs(page: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        raise LLMCallError("No LLM chunks were returned")
    if len(chunks) == 1:
        return chunks[0]

    combined = {
        "source_id": page["id"],
        "tei_xml": combine_tei_documents([chunk["tei_xml"] for chunk in chunks if chunk.get("tei_xml")]),
        "people": [],
        "locations": [],
        "claims": [],
        "event_suggestions": [],
        "quotes": [],
    }
    for chunk in chunks:
        for key in ["people", "locations", "claims", "event_suggestions", "quotes"]:
            combined[key].extend(chunk.get(key, []))
    return combined


def combine_tei_documents(tei_documents: list[str]) -> str:
    if not tei_documents:
        raise LLMCallError("LLM did not return tei_xml")
    first_root = ET.fromstring(tei_documents[0])
    first_div = first_root.find(f".//{tei_tag('body')}/{tei_tag('div')}")
    if first_div is None:
        return tei_documents[0]

    first_standoff = first_root.find(tei_tag("standOff"))
    for tei_xml in tei_documents[1:]:
        root = ET.fromstring(tei_xml)
        div = root.find(f".//{tei_tag('body')}/{tei_tag('div')}")
        if div is not None:
            for child in list(div):
                first_div.append(child)
        standoff = root.find(tei_tag("standOff"))
        if standoff is not None:
            if first_standoff is None:
                first_standoff = ET.SubElement(first_root, tei_tag("standOff"))
            for child in list(standoff):
                first_standoff.append(child)
    return serialize_xml(first_root)


def call_openai_structured(model: str, input_messages: list[dict[str, str]]) -> tuple[dict[str, Any], Any, dict[str, int]]:
    from openai import OpenAI

    schema = load_openai_extraction_schema()
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=input_messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "haymarket_extraction_bundle",
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
        raise LLMCallError("OpenAI returned output that could not be parsed as JSON", raw_output, usage) from exc
    return parsed, raw_output, usage


def load_openai_extraction_schema() -> dict[str, Any]:
    schema = json.loads((SCHEMA_DIR / "extraction_bundle.schema.json").read_text(encoding="utf-8"))
    refs = {
        "person.schema.json": json.loads((SCHEMA_DIR / "person.schema.json").read_text(encoding="utf-8")),
        "location.schema.json": json.loads((SCHEMA_DIR / "location.schema.json").read_text(encoding="utf-8")),
        "claim.schema.json": json.loads((SCHEMA_DIR / "claim.schema.json").read_text(encoding="utf-8")),
        "event.schema.json": json.loads((SCHEMA_DIR / "event.schema.json").read_text(encoding="utf-8")),
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


def estimate_cost_usd(model: str, usage: dict[str, int]) -> float:
    pricing = MODEL_PRICING_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (usage["input_tokens"] / 1_000_000 * pricing["input"]) + (usage["output_tokens"] / 1_000_000 * pricing["output"])


def build_cost_summary(run_id: str, audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    by_model: dict[str, dict[str, Any]] = {}
    calls = []

    for record in audit_records:
        usage = record["usage"]
        model = record["model"]
        totals["calls"] += 1
        totals["input_tokens"] += usage["input_tokens"]
        totals["output_tokens"] += usage["output_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["cost_usd"] += record["cost_usd"]

        model_totals = by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
        model_totals["calls"] += 1
        model_totals["input_tokens"] += usage["input_tokens"]
        model_totals["output_tokens"] += usage["output_tokens"]
        model_totals["total_tokens"] += usage["total_tokens"]
        model_totals["cost_usd"] += record["cost_usd"]

        calls.append(
            {
                "call_id": record["call_id"],
                "provider": record["provider"],
                "model": model,
                "cost_usd": record["cost_usd"],
                "status": record["status"],
            }
        )

    totals["cost_usd"] = round(totals["cost_usd"], 8)
    for values in by_model.values():
        values["cost_usd"] = round(values["cost_usd"], 8)

    return {"run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(), "totals": totals, "by_model": by_model, "calls": calls}


def build_model_eval(run_id: str, bundles_by_model: dict[str, list[dict[str, Any]]], audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    models = []
    for model, bundles in bundles_by_model.items():
        records = [record for record in audit_records if record["model"] == model]
        successful = [record for record in records if record["parsed_output"]]
        cost = round(sum(record["cost_usd"] for record in records), 8)
        missing = collect_missing_required_fields(bundles)
        models.append(
            {
                "model": model,
                "pages": len(records),
                "parse_success_rate": round(len(successful) / len(records), 4) if records else 0,
                "schema_error_count": 0,
                "people_count": sum(len(bundle.get("people", [])) for bundle in bundles),
                "location_count": sum(len(bundle.get("locations", [])) for bundle in bundles),
                "claim_count": sum(len(bundle.get("claims", [])) for bundle in bundles),
                "event_suggestion_count": sum(len(bundle.get("event_suggestions", [])) for bundle in bundles),
                "quote_count": sum(len(bundle.get("quotes", [])) for bundle in bundles),
                "tei_valid_count": sum(1 for bundle in bundles if (bundle.get("tei_validation") or {}).get("status") == "valid"),
                "missing_required_fields": sorted(missing),
                "cost_usd": cost,
            }
        )
    return {"run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(), "models": models}


def collect_missing_required_fields(bundles: list[dict[str, Any]]) -> set[str]:
    required = {
        "bundle": ["tei_xml"],
        "people": ["id", "display_name", "roles", "bio"],
        "locations": ["id", "name", "address_1886", "coordinates"],
        "claims": ["id", "reported_by_person_id", "statement", "event_time"],
        "event_suggestions": ["id", "title", "time", "claim_ids"],
        "quotes": ["id", "speaker_person_id", "quote"],
    }
    missing: set[str] = set()
    for bundle in bundles:
        for field in required["bundle"]:
            if field not in bundle:
                missing.add(f"bundle.{field}")
        for collection, fields in required.items():
            if collection == "bundle":
                continue
            for item in bundle.get(collection, []):
                for field in fields:
                    if field not in item:
                        missing.add(f"{collection}.{field}")
    return missing
