from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from enrichment.harmonization import harmonize_bundles
from enrichment.llm_extraction import extract_pages_with_audit
from enrichment.schema_validation import validate_items
from sources.hadc_source import add_standoff_annotations_to_tei, tei_to_transcript_json
from utils.s3_storage import JsonStorage


def run_enrichment(
    storage: JsonStorage,
    run_id: str,
    corpus: str,
    llm_provider: str,
    llm_models: list[str],
    briefing_model: str = "gpt-4.1-mini",
    max_tagging_workers: int = 8,
    streaming: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    pages = load_pages(storage, run_id)
    extraction = extract_pages_with_audit(
        pages=pages,
        storage=storage,
        run_id=run_id,
        provider=llm_provider,
        models=llm_models,
        briefing_model=briefing_model,
        max_tagging_workers=max_tagging_workers,
        streaming=streaming,
    )
    failed_calls = [record for record in extraction["audit_records"] if record["status"] == "error"]
    successful_calls = [record for record in extraction["audit_records"] if record["status"] == "success"]
    if failed_calls:
        print(f"LLM extraction errors: {len(failed_calls)} of {len(extraction['audit_records'])} calls failed")
        for record in failed_calls[:3]:
            print(f"- {record['call_id']}: {record['error']}")
    if not successful_calls:
        raise RuntimeError("All LLM extraction calls failed; see raw/haymarket/llm audit files for details.")

    print(f"LLM extraction successes: {len(successful_calls)} of {len(extraction['audit_records'])} calls")
    print_cost_summary(extraction["cost_summary"])
    print_model_eval(extraction["model_eval"])

    selected_model = choose_output_model(extraction["model_eval"])
    print(f"Selected model for app-ready output: {selected_model}")
    bundles = extraction["bundles_by_model"].get(selected_model, [])
    harmonized = harmonize_bundles(bundles)
    people = harmonized["people"]
    locations = harmonized["locations"]
    claims = enrich_claims_with_source(harmonized["claims"], pages)
    events = [event for event in harmonized["events"] if is_historical_event(event)]
    quotes = harmonized["quotes"]
    sources = source_summaries(pages)
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


def print_cost_summary(cost_summary: dict[str, Any]) -> None:
    totals = cost_summary.get("totals", {})
    print(
        f"LLM cost totals: calls={totals.get('calls', 0)}, "
        f"input_tokens={totals.get('input_tokens', 0)}, "
        f"output_tokens={totals.get('output_tokens', 0)}, "
        f"cost=${totals.get('cost_usd', 0.0):.6f}"
    )
    by_stage = cost_summary.get("by_stage") or {}
    for stage, values in by_stage.items():
        print(
            f"  stage {stage}: calls={values.get('calls', 0)}, "
            f"input_tokens={values.get('input_tokens', 0)}, "
            f"output_tokens={values.get('output_tokens', 0)}, "
            f"cost=${values.get('cost_usd', 0.0):.6f}"
        )


def print_model_eval(model_eval: dict[str, Any]) -> None:
    for model in model_eval.get("models", []):
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


def source_summaries(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": page["id"],
            "url": page["url"],
            "title": page["title"],
            "source_type": page["source_type"],
            "transcript_metadata": page.get("transcript_metadata", {}),
            "candidate_text_path": page.get("candidate_text_path"),
            "tei_path": page.get("tei_path"),
            "transcript_json_path": page.get("transcript_json_path"),
        }
        for page in pages
    ]


def write_harmonized_transcripts(storage: JsonStorage, pages: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> dict[str, int]:
    bundles_by_source = {bundle.get("source_id"): bundle for bundle in bundles if bundle.get("source_id")}
    summary = {"written": 0, "mentions": 0, "speaker_attributed_segments": 0}
    for page in pages:
        bundle = bundles_by_source.get(page["id"])
        tei_path = page.get("tei_path")
        transcript_path = page.get("transcript_json_path")
        if not bundle or not bundle.get("tei_xml") or not tei_path or not transcript_path:
            continue

        storage.write_text(tei_path, bundle["tei_xml"], "application/tei+xml; charset=utf-8")
        transcript_index = tei_to_transcript_json(
            source_id=page["id"],
            url=page["url"],
            title=page["title"],
            source_type=page["source_type"],
            fetched_at=page["fetched_at"],
            tei_xml=bundle["tei_xml"],
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
