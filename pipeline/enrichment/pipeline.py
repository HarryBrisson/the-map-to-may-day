from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from enrichment.harmonization import harmonize_bundles
from enrichment.llm_extraction import extract_pages_with_audit
from enrichment.schema_validation import validate_items
from enrichment.stage_briefing import run_briefing
from enrichment.progress import StageProgress
from sources.hadc_source import add_standoff_annotations_to_tei, tei_to_transcript_json
from utils.ids import slugify
from utils.s3_storage import JsonStorage


def run_enrichment(
    storage: JsonStorage,
    run_id: str,
    corpus: str,
    llm_provider: str,
    llm_models: list[str],
    briefing_model: str = "gpt-4.1-mini",
    tagging_model: str | None = None,
    max_tagging_workers: int = 8,
    max_tagging_unit_attempts: int = 3,
    streaming: bool = True,
    page_filter: list[str] | None = None,
    max_transcription_attempts: int = 2,
    transcription_format: str = "tei",
    use_cache: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    pages = load_pages(storage, run_id)
    if page_filter:
        before = len(pages)
        pages = [page for page in pages if any(token in page["id"] for token in page_filter)]
        print(f"Filtered pages by {page_filter}: {before} -> {len(pages)} page(s)")
        if not pages:
            raise RuntimeError(f"--pages filter {page_filter} matched 0 pages")
    extraction = extract_pages_with_audit(
        pages=pages,
        storage=storage,
        run_id=run_id,
        provider=llm_provider,
        models=llm_models,
        briefing_model=briefing_model,
        tagging_model=tagging_model,
        max_tagging_workers=max_tagging_workers,
        max_tagging_unit_attempts=max_tagging_unit_attempts,
        streaming=streaming,
        max_transcription_attempts=max_transcription_attempts,
        transcription_format=transcription_format,
        use_cache=use_cache,
    )
    failed_calls = [record for record in extraction["audit_records"] if record["status"] == "error"]
    successful_calls = [
        record for record in extraction["audit_records"] if record["status"] in ("success", "cached")
    ]
    cached_calls = [record for record in extraction["audit_records"] if record["status"] == "cached"]
    if failed_calls:
        print(f"LLM extraction errors: {len(failed_calls)} of {len(extraction['audit_records'])} calls failed")
        for record in failed_calls[:3]:
            print(f"- {record['call_id']}: {record['error']}")
    cache_note = f" ({len(cached_calls)} from cache)" if cached_calls else ""
    print(f"LLM extraction successes: {len(successful_calls)} of {len(extraction['audit_records'])} calls{cache_note}")
    print_model_eval(extraction["model_eval"])
    print_cost_summary(extraction["cost_summary"])
    if not successful_calls:
        raise RuntimeError("All LLM extraction calls failed; see raw/haymarket/llm audit files for details.")

    selected_model = choose_output_model(extraction["model_eval"])
    print(f"Selected model for app-ready output: {selected_model}")
    bundles = extraction["bundles_by_model"].get(selected_model, [])
    harmonized = harmonize_bundles(bundles)
    people = harmonized["people"]
    locations = harmonized["locations"]
    claims = enrich_claims_with_source(harmonized["claims"], pages)
    events = [event for event in harmonized["events"] if is_historical_event(event)]
    quotes = harmonized["quotes"]
    sources = source_summaries(pages, harmonized["bundles"])
    transcript_summary = write_harmonized_transcripts(storage, pages, harmonized["bundles"])
    print(
        "Harmonized "
        f"{len(people)} people, {len(locations)} locations, {len(events)} events "
        f"(merged people={harmonized['merge_counts']['people']}, "
        f"locations={harmonized['merge_counts']['locations']}, "
        f"events={harmonized['merge_counts']['events']})"
    )
    print(
        "Updated transcripts: "
        f"{transcript_summary['written']} written, "
        f"{transcript_summary['mentions']} mentions, "
        f"{transcript_summary['speaker_attributed_segments']} speaker-attributed segments"
    )

    validation_errors = []
    validation_errors.extend(validate_items(people, "person.schema.json"))
    validation_errors.extend(validate_items(locations, "location.schema.json"))
    validation_errors.extend(validate_items([strip_claim_app_fields(claim) for claim in claims], "claim.schema.json"))
    validation_errors.extend(validate_items(events, "event.schema.json"))

    manifest = {
        "run_id": run_id,
        "corpus": corpus,
        "selected_model": selected_model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "people": len(people),
            "locations": len(locations),
            "claims": len(claims),
            "events": len(events),
            "sources": len(sources),
            "quotes": len(quotes),
        },
        "entity_id_maps": harmonized["id_maps"],
        "transcripts": transcript_summary,
        "validation_errors": validation_errors,
    }
    if validation_errors:
        print(f"Validation errors: {len(validation_errors)}")
    else:
        print("Validation errors: 0")

    storage.write_json("enriched/haymarket/people/latest.json", people)
    storage.write_json("enriched/haymarket/locations/latest.json", locations)
    storage.write_json("enriched/haymarket/claims/latest.json", claims)
    storage.write_json("enriched/haymarket/events/latest.json", events)
    storage.write_json("enriched/haymarket/sources/latest.json", sources)
    storage.write_json("enriched/haymarket/manifest/latest.json", manifest)

    return {
        "people": people,
        "locations": locations,
        "claims": claims,
        "events": events,
        "quotes": quotes,
        "manifest": manifest,
        "model_eval": extraction["model_eval"],
        "cost_summary": extraction["cost_summary"],
    }


def run_brief_update(
    storage: JsonStorage,
    run_id: str,
    corpus: str,
    briefing_model: str = "gpt-5-mini",
    streaming: bool = True,
    page_filter: list[str] | None = None,
    resume: bool = True,
    max_brief_workers: int = 1,
) -> dict[str, Any]:
    pages = load_pages(storage, run_id)
    if page_filter:
        before = len(pages)
        pages = [page for page in pages if any(token in page["id"] for token in page_filter)]
        print(f"Filtered pages by {page_filter}: {before} -> {len(pages)} page(s)")
        if not pages:
            raise RuntimeError(f"--pages filter {page_filter} matched 0 pages")

    existing_people = read_dataset_if_available(storage, "enriched/haymarket/people/latest.json")
    existing_locations = read_dataset_if_available(storage, "enriched/haymarket/locations/latest.json")
    existing_claims = read_dataset_if_available(storage, "enriched/haymarket/claims/latest.json")
    existing_events = read_dataset_if_available(storage, "enriched/haymarket/events/latest.json")

    briefings_by_source: dict[str, dict[str, Any]] = {}
    audit_records: list[dict[str, Any]] = []
    total_cost = 0.0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "cached_input_tokens": 0}
    reused = 0
    pages_to_brief = [page for page in pages if page.get("source_type") != "toc"]
    for page in pages:
        if page.get("source_type") == "toc":
            print(f"Skipping brief update for {page['id']} (source_type=toc)")

    def brief_page(page: dict[str, Any]) -> dict[str, Any]:
        cached_audit = read_successful_brief_audit(storage, run_id, briefing_model, page["id"]) if resume else None
        if cached_audit:
            result = briefing_result_from_audit(cached_audit, run_id, briefing_model, page["id"])
            StageProgress(page["id"], "briefing", enabled=streaming).info("reused existing successful brief")
            result["reused"] = True
            return result
        result = run_briefing(
            page=page,
            storage=storage,
            run_id=run_id,
            model=briefing_model,
            progress=StageProgress(page["id"], "briefing", enabled=streaming),
        )
        result["reused"] = False
        return result

    max_workers = max(1, int(max_brief_workers or 1))
    if max_workers > 1:
        print(f"Brief update workers: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(brief_page, page): page for page in pages_to_brief}
        for future in as_completed(futures):
            page = futures[future]
            result = future.result()
            if result.get("reused"):
                reused += 1
            if result.get("audit") is None:
                raise RuntimeError(f"Briefing produced no audit record for {page['id']}")
            audit_records.append(result["audit"])
            total_cost += float(result.get("cost_usd") or 0.0)
            for key in total_usage:
                total_usage[key] += (result.get("usage") or {}).get(key, 0)
            if result["status"] in {"success", "cached"}:
                briefings_by_source[page["id"]] = result["briefing"] or {}

    bundles = [
        brief_navigation_bundle(
            page=page,
            briefing=briefings_by_source.get(page["id"]),
            people=existing_people,
            locations=existing_locations,
            claims=existing_claims,
            events=existing_events,
        )
        for page in pages
    ]
    sources = source_summaries(pages, bundles)
    storage.write_json("enriched/haymarket/sources/latest.json", sources)

    manifest = read_json_if_available(storage, "enriched/haymarket/manifest/latest.json") or {}
    manifest.update(
        {
            "run_id": manifest.get("run_id") or run_id,
            "corpus": manifest.get("corpus") or corpus,
            "sources_briefed_at": datetime.now(timezone.utc).isoformat(),
            "sources_brief_run_id": run_id,
            "sources_brief_model": briefing_model,
        }
    )
    counts = dict(manifest.get("counts") or {})
    counts["sources"] = len(sources)
    manifest["counts"] = counts
    storage.write_json("enriched/haymarket/manifest/latest.json", manifest)

    return {
        "sources": sources,
        "briefings": len(briefings_by_source),
        "pages": len(pages_to_brief),
        "skipped": len(pages) - len(pages_to_brief),
        "reused": reused,
        "audit_records": audit_records,
        "cost_usd": round(total_cost, 8),
        "usage": total_usage,
    }


def read_successful_brief_audit(
    storage: JsonStorage,
    run_id: str,
    model: str,
    page_id: str,
) -> dict[str, Any] | None:
    audit_path = brief_audit_path(run_id, model, page_id)
    if not storage.exists(audit_path):
        return None
    audit = storage.read_json(audit_path)
    if audit.get("status") != "success" or not audit.get("parsed_output"):
        return None
    return audit


def briefing_result_from_audit(audit: dict[str, Any], run_id: str, model: str, page_id: str) -> dict[str, Any]:
    return {
        "briefing": audit.get("parsed_output") or {},
        "audit": audit,
        "usage": zero_usage(),
        "cost_usd": 0.0,
        "status": "cached",
        "error": None,
        "audit_path": brief_audit_path(run_id, model, page_id),
    }


def brief_audit_path(run_id: str, model: str, page_id: str) -> str:
    return f"raw/haymarket/llm/{run_id}/{slugify(model)}/{page_id}/briefing.json"


def zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "cached_input_tokens": 0}


def brief_navigation_bundle(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    source_id = page["id"]
    source_claims = [claim for claim in claims if claim.get("source_id") == source_id]
    source_claim_ids = {claim.get("id") for claim in source_claims}
    return {
        "source_id": source_id,
        "briefing": briefing,
        "people": [person for person in people if source_id in person.get("source_ids", [])],
        "all_people": people,
        "locations": [location for location in locations if source_id in location.get("source_ids", [])],
        "all_locations": locations,
        "claims": source_claims,
        "event_suggestions": [
            event
            for event in events
            if source_claim_ids.intersection(event.get("claim_ids", []))
        ],
        "all_events": events,
        "quotes": [],
    }


def read_dataset_if_available(storage: JsonStorage, relative_path: str) -> list[dict[str, Any]]:
    data = read_json_if_available(storage, relative_path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []


def read_json_if_available(storage: JsonStorage, relative_path: str) -> Any:
    if not storage.exists(relative_path):
        return None
    return storage.read_json(relative_path)


def print_cost_summary(cost_summary: dict[str, Any]) -> None:
    totals = cost_summary.get("totals", {})
    reasoning = totals.get("reasoning_tokens", 0)
    reasoning_suffix = f", {reasoning} reasoning" if reasoning else ""
    print(
        "LLM cost: "
        f"${totals.get('cost_usd', 0):.6f} total across {totals.get('calls', 0)} calls "
        f"({totals.get('input_tokens', 0)} input + {totals.get('output_tokens', 0)} "
        f"output tokens{reasoning_suffix})"
    )
    for model, values in cost_summary.get("by_model", {}).items():
        print(f"  {model}: ${values.get('cost_usd', 0):.6f} ({values.get('calls', 0)} calls)")
    for stage, values in cost_summary.get("by_stage", {}).items():
        stage_reasoning = values.get("reasoning_tokens", 0)
        stage_reasoning_suffix = f", {stage_reasoning} reasoning" if stage_reasoning else ""
        print(
            f"  stage {stage}: ${values.get('cost_usd', 0):.6f} "
            f"({values.get('calls', 0)} calls, "
            f"{values.get('input_tokens', 0)} input + {values.get('output_tokens', 0)} "
            f"output tokens{stage_reasoning_suffix})"
        )


def print_model_eval(model_eval: dict[str, Any]) -> None:
    models = model_eval.get("models", [])
    if not models:
        return
    if len(models) == 1:
        model = models[0]
        print(
            "Model "
            f"{model['model']}: parse_success={model['parse_success_rate']:.0%}, "
            f"people={model['people_count']}, "
            f"locations={model['location_count']}, "
            f"claims={model['claim_count']}, "
            f"events={model['event_suggestion_count']}, "
            f"quotes={model.get('quote_count', 0)}, "
            f"tei_valid={model.get('tei_valid_count', 0)}/{model['pages']}, "
            f"cost=${model['cost_usd']:.6f}"
        )
        return

    headers = ["model", "pages", "valid", "people", "locs", "claims", "events", "quotes", "cost"]
    rows = [
        [
            model["model"],
            str(model["pages"]),
            f"{model.get('tei_valid_count', 0)}/{model['pages']}",
            str(model["people_count"]),
            str(model["location_count"]),
            str(model["claim_count"]),
            str(model["event_suggestion_count"]),
            str(model.get("quote_count", 0)),
            f"${model['cost_usd']:.4f}",
        ]
        for model in models
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print("Model comparison:")
    print("  " + format_row(headers))
    print("  " + format_row(["-" * w for w in widths]))
    for row in rows:
        print("  " + format_row(row))


def load_pages(storage: JsonStorage, run_id: str) -> list[dict[str, Any]]:
    page_path = f"raw/haymarket/hadc/{run_id}/pages.json"
    if storage.exists(page_path):
        return storage.read_json(page_path)

    latest_run = storage.read_json("raw/haymarket/hadc/latest_run.json")
    return storage.read_json(f"raw/haymarket/hadc/{latest_run['run_id']}/pages.json")


def choose_output_model(model_eval: dict[str, Any]) -> str:
    models = model_eval.get("models", [])
    if not models:
        raise RuntimeError("No model evaluation data was produced")

    ranked = sorted(
        models,
        key=lambda item: (
            item.get("parse_success_rate", 0),
            -item.get("schema_error_count", 0),
            item.get("claim_count", 0),
            item.get("people_count", 0),
        ),
        reverse=True,
    )
    return ranked[0]["model"]


def collect(bundles: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for bundle in bundles:
        items.extend(bundle.get(key, []))
    return items


def merge_by_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("id")
        if not item_id:
            continue
        if item_id not in merged:
            merged[item_id] = dict(item)
            continue
        merged[item_id] = merge_item(merged[item_id], item)
    return sorted(merged.values(), key=lambda item: item["id"])


def merge_item(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(existing)
    for key, value in incoming.items():
        if value in (None, "", []):
            continue
        current = result.get(key)
        if isinstance(current, list) and isinstance(value, list):
            result[key] = sorted({*current, *value})
        elif isinstance(current, dict) and isinstance(value, dict):
            result[key] = merge_item(current, value)
        elif key == "confidence" and isinstance(value, (int, float)):
            result[key] = max(float(current or 0), float(value))
        elif current in (None, "", []):
            result[key] = value
    return result


def merge_people(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_by_id(people)


def merge_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_by_id(locations)


def merge_events(events: list[dict[str, Any]], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if events:
        return merge_by_id([event for event in events if is_historical_event(event)])

    return []


def is_historical_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "").strip().lower()
    title = str(event.get("title") or "").strip().lower()
    excluded_types = {"evidence", "exhibit", "testimony", "source", "document"}
    excluded_title_terms = ["introduced into evidence", "diagram of", "testimony of"]
    if event_type in excluded_types:
        return False
    return not any(term in title for term in excluded_title_terms)


def enrich_claims_with_source(claims: list[dict[str, Any]], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages_by_id = {page["id"]: page for page in pages}
    enriched = []
    for claim in claims:
        page = pages_by_id.get(claim.get("source_id"))
        app_claim = dict(claim)
        if page:
            app_claim["source"] = {
                "id": page["id"],
                "title": page["title"],
                "url": page["url"],
                "metadata": page.get("transcript_metadata", {}),
            }
        enriched.append(app_claim)
    return enriched


def strip_claim_app_fields(claim: dict[str, Any]) -> dict[str, Any]:
    schema_claim = dict(claim)
    schema_claim.pop("source", None)
    return schema_claim


def source_summaries(pages: list[dict[str, Any]], bundles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    bundles_by_source = {bundle.get("source_id"): bundle for bundle in bundles or [] if bundle.get("source_id")}
    return [
        {
            "id": page["id"],
            "url": page["url"],
            "title": page["title"],
            "source_type": page["source_type"],
            "transcript_metadata": page.get("transcript_metadata", {}),
            "source_stats": page.get("source_stats", {}),
            "candidate_text_path": page.get("candidate_text_path"),
            "tei_path": page.get("tei_path"),
            "transcript_json_path": page.get("transcript_json_path"),
            "navigation": build_source_navigation(page, bundles_by_source.get(page["id"])),
        }
        for page in pages
    ]


def build_source_navigation(page: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    briefing = (bundle or {}).get("briefing") or {}
    metadata = page.get("transcript_metadata", {}) or {}
    people = (bundle or {}).get("people", [])
    locations = (bundle or {}).get("locations", [])
    events = (bundle or {}).get("event_suggestions", [])
    all_people = merge_entity_pool(people, (bundle or {}).get("all_people", []))
    all_locations = merge_entity_pool(locations, (bundle or {}).get("all_locations", []))
    all_events = merge_entity_pool(events, (bundle or {}).get("all_events", []))

    document_date = normalize_document_date(briefing.get("document_date"), metadata)
    document_order = normalize_document_order(briefing.get("document_order"), metadata)
    primary_people = normalize_entity_refs(
        briefing.get("primary_people") or [],
        all_people,
        id_key="id",
        label_keys=("display_name", "alternate_names"),
    )
    if not primary_people:
        primary_people = [
            {
                "label": person.get("display_name") or person.get("id"),
                "canonical_id": person.get("id"),
                "role_or_relationship": ", ".join(person.get("roles", [])[:2]) or None,
                "confidence": person.get("confidence"),
            }
            for person in people[:8]
            if person.get("id") or person.get("display_name")
        ]

    primary_locations = normalize_entity_refs(
        briefing.get("primary_locations") or [],
        all_locations,
        id_key="id",
        label_keys=("name", "address_1886", "address_1887", "modern_address"),
    )
    if not primary_locations:
        primary_locations = [
            {
                "label": location.get("name") or location.get("id"),
                "canonical_id": location.get("id"),
                "role_or_relationship": location.get("location_type"),
                "confidence": location.get("confidence"),
            }
            for location in locations[:8]
            if location.get("id") or location.get("name")
        ]

    raw_referenced_events = briefing.get("referenced_events") or []
    raw_document_events = briefing.get("document_events") or []
    if not raw_document_events:
        raw_referenced_events, raw_document_events = split_document_event_refs(raw_referenced_events)

    referenced_events = normalize_event_refs(
        raw_referenced_events,
        all_events,
        all_people,
        all_locations,
    )
    if not referenced_events:
        referenced_events = [event_to_navigation_ref(event, people, locations) for event in events[:8]]
    document_events = normalize_document_event_refs(
        raw_document_events,
        all_people,
        all_locations,
    )

    return {
        "brief_title": briefing.get("brief_title") or page.get("title"),
        "navigation_summary": briefing.get("navigation_summary") or briefing.get("summary") or "",
        "document_date": document_date,
        "document_order": document_order,
        "document_role": briefing.get("document_role") or infer_document_role(page),
        "topics": sorted({str(topic).strip() for topic in briefing.get("topics", []) if str(topic).strip()}),
        "primary_people": primary_people,
        "primary_locations": primary_locations,
        "referenced_events": referenced_events,
        "document_events": document_events,
        "claim_count": len((bundle or {}).get("claims", [])),
        "event_reference_count": len(referenced_events),
        "document_event_count": len(document_events),
        "confidence": average_confidence([*primary_people, *primary_locations, *referenced_events, *document_events]),
    }


def normalize_document_date(value: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    original_text = source.get("original_text") or metadata.get("date_text")
    normalized = source.get("normalized_date") or parse_date_text(original_text)
    return {
        "original_text": original_text,
        "normalized_date": normalized,
        "precision": source.get("precision") or ("day" if normalized else "unknown"),
    }


def normalize_document_order(value: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    page_start, page_end = parse_page_range(metadata.get("pages"))
    return {
        "volume": metadata.get("volume") or source.get("volume"),
        "page_start": page_start if page_start is not None else source.get("page_start"),
        "page_end": page_end if page_end is not None else source.get("page_end"),
        "sequence_label": metadata.get("pages") if page_start is not None else source.get("sequence_label"),
    }


def merge_entity_pool(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in [*primary, *secondary]:
        entity_id = entity.get("id")
        if entity_id and entity_id in seen:
            continue
        if entity_id:
            seen.add(entity_id)
        merged.append(entity)
    return merged


def normalize_entity_refs(
    refs: list[Any],
    entities: list[dict[str, Any]],
    *,
    id_key: str,
    label_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    normalized = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        label = ref.get("label")
        canonical_id = ref.get("canonical_id")
        entity = find_entity(canonical_id, label, entities, id_key=id_key, label_keys=label_keys)
        normalized.append(
            {
                "label": entity_label(entity, label_keys) if entity else label,
                "canonical_id": entity.get(id_key) if entity else canonical_id,
                "role_or_relationship": ref.get("role_or_relationship"),
                "confidence": ref.get("confidence"),
            }
        )
    return [item for item in normalized if item.get("label") or item.get("canonical_id")]


def normalize_event_refs(
    refs: list[Any],
    events: list[dict[str, Any]],
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        event = find_entity(ref.get("canonical_id"), ref.get("label"), events, id_key="id", label_keys=("title",))
        if event:
            item = event_to_navigation_ref(event, people, locations)
            item["supporting_quote"] = ref.get("supporting_quote")
            item["page_refs"] = normalize_page_refs(ref.get("page_refs") or item["page_refs"])
            normalized.append(item)
            continue
        location = find_entity(ref.get("location_id"), ref.get("location_label"), locations, id_key="id", label_keys=("name", "address_1886", "address_1887", "modern_address"))
        participant_ids = normalize_participant_ids(ref, people)
        normalized.append(
            {
                "label": ref.get("label"),
                "canonical_id": ref.get("canonical_id"),
                "event_time": event_time_to_ref(ref.get("event_time") or {}),
                "location_label": entity_label(location, ("name",)) if location else ref.get("location_label"),
                "location_id": location.get("id") if location else ref.get("location_id"),
                "participant_labels": ref.get("participant_labels") or [],
                "participant_person_ids": participant_ids,
                "summary": ref.get("summary") or "",
                "supporting_quote": ref.get("supporting_quote"),
                "page_refs": normalize_page_refs(ref.get("page_refs") or []),
                "confidence": ref.get("confidence"),
            }
        )
    return [item for item in normalized if item.get("label")]


def normalize_document_event_refs(
    refs: list[Any],
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        location = find_entity(ref.get("location_id"), ref.get("location_label"), locations, id_key="id", label_keys=("name", "address_1886", "address_1887", "modern_address"))
        normalized.append(
            {
                "label": ref.get("label"),
                "event_kind": ref.get("event_kind") or infer_document_event_kind(ref),
                "event_time": event_time_to_ref(ref.get("event_time") or {}),
                "location_label": entity_label(location, ("name",)) if location else ref.get("location_label"),
                "location_id": location.get("id") if location else ref.get("location_id"),
                "participant_labels": ref.get("participant_labels") or [],
                "participant_person_ids": normalize_participant_ids(ref, people),
                "summary": ref.get("summary") or "",
                "supporting_quote": ref.get("supporting_quote"),
                "page_refs": normalize_page_refs(ref.get("page_refs") or []),
                "confidence": ref.get("confidence"),
            }
        )
    return [item for item in normalized if item.get("label")]


def split_document_event_refs(refs: list[Any]) -> tuple[list[Any], list[Any]]:
    historical_refs: list[Any] = []
    document_refs: list[Any] = []
    for ref in refs:
        if isinstance(ref, dict) and is_document_event_ref(ref):
            document_refs.append(ref)
        else:
            historical_refs.append(ref)
    return historical_refs, document_refs


def is_document_event_ref(ref: dict[str, Any]) -> bool:
    text = normalize_label(" ".join(str(ref.get(key) or "") for key in ("label", "summary")))
    document_terms = (
        "introduced into evidence",
        "introduction of",
        "publication",
        "published",
        "filing",
        "filed",
        "catalog",
        "page copying",
        "transcript",
        "change of venue",
        "petition",
        "motion",
        "order",
        "arraignment",
        "remand",
        "grand jury presentment",
        "empaneling",
        "swearing of the cook county grand jury",
    )
    return any(term in text for term in document_terms)


def infer_document_event_kind(ref: dict[str, Any]) -> str:
    text = normalize_label(" ".join(str(ref.get(key) or "") for key in ("label", "summary")))
    if any(term in text for term in ("publication", "published")):
        return "publication"
    if any(term in text for term in ("filing", "filed")):
        return "filing"
    if any(term in text for term in ("introduced into evidence", "introduction of", "exhibit")):
        return "evidence_introduction"
    if any(term in text for term in ("catalog", "page copying", "source", "transcript")):
        return "source_description"
    if any(term in text for term in ("petition", "motion", "order", "arraignment", "remand", "grand jury", "empaneling")):
        return "court_procedure"
    if any(term in text for term in ("created", "creation", "drawn", "made")):
        return "document_creation"
    return "other_document_event"


def normalize_participant_ids(ref: dict[str, Any], people: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for person_id in ref.get("participant_person_ids") or []:
        if isinstance(person_id, str) and person_id and person_id not in ids:
            ids.append(person_id)
    for label in ref.get("participant_labels") or []:
        person = find_entity(None, label, people, id_key="id", label_keys=("display_name", "alternate_names"))
        person_id = person.get("id") if person else None
        if person_id and person_id not in ids:
            ids.append(person_id)
    return ids


def event_to_navigation_ref(
    event: dict[str, Any],
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> dict[str, Any]:
    people_by_id = {person.get("id"): person for person in people}
    locations_by_id = {location.get("id"): location for location in locations}
    location = locations_by_id.get(event.get("location_id"))
    participant_ids = event.get("participant_person_ids", [])
    return {
        "label": event.get("title"),
        "canonical_id": event.get("id"),
        "event_time": event_time_to_ref(event.get("time") or {}),
        "location_label": location.get("name") if location else None,
        "location_id": event.get("location_id"),
        "participant_labels": [
            (people_by_id.get(person_id) or {}).get("display_name") or person_id
            for person_id in participant_ids
        ],
        "participant_person_ids": participant_ids,
        "summary": event.get("description") or "",
        "supporting_quote": None,
        "page_refs": [],
        "confidence": event.get("confidence"),
    }


def event_time_to_ref(value: dict[str, Any]) -> dict[str, Any]:
    start = value.get("start") or value.get("normalized_date")
    return {
        "start": start,
        "end": value.get("end"),
        "normalized_date": str(start)[:10] if start else None,
        "precision": value.get("precision") or "unknown",
        "original_text": value.get("display") or value.get("original_text"),
    }


def empty_event_time() -> dict[str, Any]:
    return {"start": None, "end": None, "normalized_date": None, "precision": "unknown", "original_text": None}


def normalize_page_refs(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        for page_ref in extract_page_refs(value):
            if page_ref not in normalized:
                normalized.append(page_ref)
    return normalized


def extract_page_refs(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    refs = [
        normalize_page_ref(match.group(1))
        for match in re.finditer(
            r"\b(?:pp?\.?|pages?|page)\s*([A-Z]?\s?\d+(?:\s+1/2)?)(?:\s*[-–]\s*[A-Z]?\s?\d+(?:\s+1/2)?)?",
            text,
            flags=re.I,
        )
    ]
    if refs:
        return refs
    range_match = re.fullmatch(r"([A-Z]?\s?\d+(?:\s+1/2)?)\s*[-–]\s*[A-Z]?\s?\d+(?:\s+1/2)?", text, flags=re.I)
    if range_match:
        return [normalize_page_ref(range_match.group(1))]
    if re.fullmatch(r"[A-Z]?\s?\d+(?:\s+1/2)?", text, flags=re.I):
        return [normalize_page_ref(text)]
    return [normalize_page_ref(text)]


def normalize_page_ref(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def find_entity(
    canonical_id: Any,
    label: Any,
    entities: list[dict[str, Any]],
    *,
    id_key: str,
    label_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    if canonical_id:
        match = next((entity for entity in entities if entity.get(id_key) == canonical_id), None)
        if match:
            return match
    normalized_label = normalize_label(str(label or ""))
    if not normalized_label:
        return None
    normalized_variants = label_variants(normalized_label)
    for entity in entities:
        labels: list[Any] = []
        for key in label_keys:
            value = entity.get(key)
            labels.extend(value if isinstance(value, list) else [value])
        if any(label_variants(normalize_label(str(candidate or ""))).intersection(normalized_variants) for candidate in labels):
            return entity
    return None


def entity_label(entity: dict[str, Any], label_keys: tuple[str, ...]) -> str | None:
    for key in label_keys:
        value = entity.get(key)
        if isinstance(value, list):
            value = next((item for item in value if item), None)
        if value:
            return str(value)
    return None


def average_confidence(items: list[dict[str, Any]]) -> float | None:
    values = [float(item["confidence"]) for item in items if isinstance(item.get("confidence"), (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def infer_document_role(page: dict[str, Any]) -> str:
    source_type = page.get("source_type")
    if source_type in {"testimony", "exhibit", "toc"}:
        return source_type
    title = str(page.get("title") or "").lower()
    if "cover page" in title:
        return "cover"
    if any(term in title for term in ("summons", "motion", "order", "indictment")):
        return "legal_document"
    return "other"


def parse_page_range(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    range_match = re.search(r"([A-Z]?\s?\d+)(?:\s+1/2)?\s*[-–]\s*([A-Z]?\s?\d+)(?:\s+1/2)?", text, flags=re.I)
    if range_match:
        return page_ref_number(range_match.group(1)), page_ref_number(range_match.group(2))
    single_match = re.fullmatch(r"[A-Z]?\s?\d+(?:\s+1/2)?", text, flags=re.I)
    if single_match:
        page = page_ref_number(text)
        return page, page
    return None, None


def page_ref_number(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def label_variants(value: str) -> set[str]:
    variants = {value}
    without_parenthetical = re.sub(r"\s*\([^)]*\)", "", value).strip()
    if without_parenthetical:
        variants.add(without_parenthetical)
    return variants


def parse_date_text(value: Any) -> str | None:
    text = str(value or "").strip().replace("Sept.", "Sep.").replace("August", "Aug.")
    if not text:
        return None
    patterns = [
        ("%Y %B %d", r"(\d{4})\s+([A-Za-z]+)\.?\s+(\d{1,2})"),
        ("%Y %b %d", r"(\d{4})\s+([A-Za-z]+)\.?\s+(\d{1,2})"),
        ("%B %d, %Y", r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})"),
        ("%b %d, %Y", r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})"),
    ]
    for fmt, pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = " ".join(match.groups()) if fmt.startswith("%Y") else f"{match.group(1)} {match.group(2)}, {match.group(3)}"
        try:
            return datetime.strptime(candidate.replace(".", ""), fmt.replace(".", "")).date().isoformat()
        except ValueError:
            continue
    return None


def write_harmonized_transcripts(storage: JsonStorage, pages: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> dict[str, int]:
    bundles_by_source = {bundle.get("source_id"): bundle for bundle in bundles if bundle.get("source_id")}
    summary = {"written": 0, "mentions": 0, "speaker_attributed_segments": 0}
    for page in pages:
        bundle = bundles_by_source.get(page["id"])
        tei_path = page.get("tei_path")
        transcript_path = page.get("transcript_json_path")
        if not bundle or not bundle.get("tei_xml") or not tei_path or not transcript_path:
            continue

        # First pass: render the bundle TEI to extract plain text so we can
        # locate every mention of an entity's display_name / alternate_names
        # in the transcript.
        base_transcript = tei_to_transcript_json(
            source_id=page["id"],
            url=page["url"],
            title=page["title"],
            source_type=page["source_type"],
            fetched_at=page["fetched_at"],
            tei_xml=bundle["tei_xml"],
            transcript_metadata=page.get("transcript_metadata", {}),
        )
        # Find literal mentions in the text and add them as standoff
        # annotations on the TEI. Without this the app sees segments and
        # speaker attribution but no entity-to-text linkage.
        mentions = build_annotation_mentions(base_transcript["text"], bundle)
        annotated_tei = add_standoff_annotations_to_tei(bundle["tei_xml"], mentions)
        storage.write_text(tei_path, annotated_tei, "application/tei+xml; charset=utf-8")

        # Second pass: re-render the transcript index from the annotated TEI
        # so the published transcript JSON includes the inline mentions.
        transcript_index = tei_to_transcript_json(
            source_id=page["id"],
            url=page["url"],
            title=page["title"],
            source_type=page["source_type"],
            fetched_at=page["fetched_at"],
            tei_xml=annotated_tei,
            transcript_metadata=page.get("transcript_metadata", {}),
        )
        storage.write_json(transcript_path, transcript_index)
        summary["written"] += 1
        summary["mentions"] += len(transcript_index.get("mentions", []))
        summary["speaker_attributed_segments"] += sum(1 for segment in transcript_index.get("segments", []) if segment.get("speaker_id"))
        print(
            f"Transcript {page['id']}: segments={len(transcript_index.get('segments', []))}, "
            f"mentions={len(transcript_index.get('mentions', []))}, "
            f"speaker_attributed_segments={sum(1 for segment in transcript_index.get('segments', []) if segment.get('speaker_id'))}"
        )
    return summary


def write_transcript_annotations(storage: JsonStorage, pages: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> None:
    bundles_by_source = {bundle.get("source_id"): bundle for bundle in bundles if bundle.get("source_id")}
    for page in pages:
        tei_path = page.get("tei_path")
        transcript_path = page.get("transcript_json_path")
        bundle = bundles_by_source.get(page["id"])
        if not tei_path or not transcript_path or not bundle:
            continue

        tei_xml = storage.read_text(tei_path)
        base_transcript = tei_to_transcript_json(
            source_id=page["id"],
            url=page["url"],
            title=page["title"],
            source_type=page["source_type"],
            fetched_at=page["fetched_at"],
            tei_xml=tei_xml,
            transcript_metadata=page.get("transcript_metadata", {}),
        )
        mentions = build_annotation_mentions(base_transcript["text"], bundle)
        annotated_tei = add_standoff_annotations_to_tei(tei_xml, mentions)
        storage.write_text(tei_path, annotated_tei, "application/tei+xml; charset=utf-8")
        transcript_index = tei_to_transcript_json(
            source_id=page["id"],
            url=page["url"],
            title=page["title"],
            source_type=page["source_type"],
            fetched_at=page["fetched_at"],
            tei_xml=annotated_tei,
            transcript_metadata=page.get("transcript_metadata", {}),
        )
        storage.write_json(transcript_path, transcript_index)


def build_annotation_mentions(text: str, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for person in bundle.get("people", []):
        labels = [person.get("display_name"), *person.get("alternate_names", [])]
        candidates.extend(build_entity_candidates("person", person.get("id"), labels, person.get("confidence")))

    for location in bundle.get("locations", []):
        labels = [location.get("name"), location.get("address_1886"), location.get("address_1887")]
        candidates.extend(build_entity_candidates("location", location.get("id"), labels, location.get("confidence")))

    for claim in bundle.get("claims", []):
        labels = [claim.get("quote")]
        candidates.extend(build_entity_candidates("claim", claim.get("id"), labels, claim.get("confidence")))

    for event in bundle.get("event_suggestions", []):
        labels = [event.get("title")]
        candidates.extend(build_entity_candidates("event", event.get("id"), labels, event.get("confidence")))

    mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for candidate in candidates:
        label = str(candidate["label"]).strip()
        if len(label) < 3:
            continue
        for match in re_finditer_literal(label, text):
            key = (candidate["kind"], candidate["entity_id"], match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                {
                    "id": f"mention_{candidate['kind']}_{candidate['entity_id']}_{match.start()}_{match.end()}",
                    "kind": candidate["kind"],
                    "entity_id": candidate["entity_id"],
                    "start": match.start(),
                    "end": match.end(),
                    "text": text[match.start() : match.end()],
                    "confidence": candidate.get("confidence"),
                    "source": "llm_extraction",
                }
            )
    return sorted(mentions, key=lambda item: (item["start"], item["end"], item["kind"]))


def build_entity_candidates(kind: str, entity_id: str | None, labels: list[Any], confidence: Any) -> list[dict[str, Any]]:
    if not entity_id:
        return []
    return [
        {
            "kind": kind,
            "entity_id": entity_id,
            "label": label,
            "confidence": confidence if isinstance(confidence, (int, float)) else None,
        }
        for label in labels
        if label
    ]


def re_finditer_literal(needle: str, haystack: str):
    import re

    return re.finditer(re.escape(needle), haystack, flags=re.IGNORECASE)
