from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from utils.ids import slugify
from utils.openai_schema import call_openai_embeddings, call_openai_structured, estimate_cost_usd
from utils.s3_storage import JsonStorage


HARMONIZATION_VERSION = "brief_harmonization_v2"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_REVIEW_MODEL = "gpt-5.4-mini"
DEFAULT_ESCALATION_REVIEW_MODEL = DEFAULT_REVIEW_MODEL
DEFAULT_REVIEW_REASONING_EFFORT = "low"
DEFAULT_ESCALATION_REASONING_EFFORT = "medium"
DEFAULT_MAX_LLM_REVIEW_BATCHES = 4
DEFAULT_MAX_LLM_REVIEW_CANDIDATES = 240
HISTORICAL_CLASS = "historical_event"
DOCUMENT_CLASS = "document_event"
BOTH_CLASS = "both"
ROUTINE_CLASS = "routine_procedural"
UNCERTAIN_CLASS = "uncertain"

HISTORICAL_TERMS = (
    "haymarket",
    "mccormick",
    "bomb",
    "bombing",
    "meeting",
    "rally",
    "speech",
    "procession",
    "march",
    "riot",
    "strike",
    "arrest",
    "search",
    "police action",
    "shooting",
    "explosion",
    "throwing",
)

DOCUMENT_TERMS = (
    "affidavit",
    "appeal",
    "arraignment",
    "certification",
    "change of venue",
    "court",
    "evidence",
    "exhibit",
    "filed",
    "filing",
    "indictment",
    "introduced",
    "introduction",
    "instruction",
    "judgment",
    "motion",
    "order",
    "petition",
    "published",
    "publication",
    "sentence",
    "sentencing",
    "transcript",
    "verdict",
    "writ",
)

ROUTINE_TERMS = (
    "adjournment",
    "adjourned",
    "recess",
    "testimony given",
    "testimony recorded",
    "witness testimony",
)

TOPIC_SYNONYMS = {
    "labor movement": "labor organizing",
    "eight hour": "eight-hour movement",
    "eight-hour": "eight-hour movement",
    "dynamite": "explosives",
    "bombs": "explosives",
    "jury": "jury proceedings",
    "juror": "jury proceedings",
    "newspaper": "press and publications",
    "publication": "press and publications",
}


def run_brief_harmonization(
    *,
    storage: JsonStorage,
    pages: list[dict[str, Any]],
    briefings_by_source: dict[str, dict[str, Any]],
    run_id: str,
    people: list[dict[str, Any]] | None = None,
    locations: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    use_embeddings: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    use_llm_review: bool = False,
    review_model: str = DEFAULT_REVIEW_MODEL,
    escalation_review_model: str = DEFAULT_ESCALATION_REVIEW_MODEL,
    max_llm_review_batches: int | None = DEFAULT_MAX_LLM_REVIEW_BATCHES,
    max_llm_review_candidates: int | None = DEFAULT_MAX_LLM_REVIEW_CANDIDATES,
    progress: bool = True,
) -> dict[str, Any]:
    result = harmonize_briefings(
        storage=storage,
        pages=pages,
        briefings_by_source=briefings_by_source,
        run_id=run_id,
        people=people or [],
        locations=locations or [],
        events=events or [],
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        use_llm_review=use_llm_review,
        review_model=review_model,
        escalation_review_model=escalation_review_model,
        max_llm_review_batches=max_llm_review_batches,
        max_llm_review_candidates=max_llm_review_candidates,
        progress=progress,
    )
    raw_prefix = f"raw/haymarket/brief_harmonization/{run_id}"
    storage.write_json("enriched/haymarket/brief_harmonization/latest.json", result["artifact"])
    storage.write_json(f"{raw_prefix}/audit.json", result["artifact"])
    storage.write_json(f"{raw_prefix}/normalized_refs.json", result["normalized_refs"])
    storage.write_json(f"{raw_prefix}/embedding_inputs.json", result["embedding_inputs"])
    storage.write_json(f"{raw_prefix}/embedding_cache_manifest.json", result["embedding_cache_manifest"])
    storage.write_json(f"{raw_prefix}/candidate_matches.json", result["candidate_matches"])
    storage.write_json(f"{raw_prefix}/proposed_clusters.json", result["proposed_clusters"])
    storage.write_json(f"{raw_prefix}/final_clusters.json", result["final_clusters"])
    storage.write_json(f"{raw_prefix}/qa_report.json", result["qa_report"])
    for batch in result["llm_review_batches"]:
        storage.write_json(f"{raw_prefix}/llm_review_batches/{batch['batch_id']}.json", batch)
    return result


def harmonize_briefings(
    *,
    storage: JsonStorage | None = None,
    pages: list[dict[str, Any]],
    briefings_by_source: dict[str, dict[str, Any]],
    run_id: str,
    people: list[dict[str, Any]] | None = None,
    locations: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    use_embeddings: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    use_llm_review: bool = False,
    review_model: str = DEFAULT_REVIEW_MODEL,
    escalation_review_model: str = DEFAULT_ESCALATION_REVIEW_MODEL,
    max_llm_review_batches: int | None = DEFAULT_MAX_LLM_REVIEW_BATCHES,
    max_llm_review_candidates: int | None = DEFAULT_MAX_LLM_REVIEW_CANDIDATES,
    progress: bool = False,
) -> dict[str, Any]:
    people = people or []
    locations = locations or []
    events = events or []
    pages_by_id = {page.get("id"): page for page in pages}
    harmonized_by_source: dict[str, dict[str, Any]] = {}
    event_refs: list[dict[str, Any]] = []
    document_event_refs: list[dict[str, Any]] = []
    routine_refs: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()

    report = progress_reporter(progress)
    raw_prefix = f"raw/haymarket/brief_harmonization/{run_id}"
    report(f"Brief harmonization: normalizing {len(briefings_by_source)} raw brief(s)")
    for source_id, raw_briefing in briefings_by_source.items():
        page = pages_by_id.get(source_id, {"id": source_id, "transcript_metadata": {}})
        harmonized = copy.deepcopy(raw_briefing or {})
        harmonized["source_id"] = harmonized.get("source_id") or source_id
        harmonized["brief_title"] = clean_text(harmonized.get("brief_title")) or page.get("title") or source_id
        harmonized["navigation_summary"] = clean_text(
            harmonized.get("navigation_summary") or harmonized.get("summary")
        )
        harmonized["topics"] = normalize_topics(harmonized.get("topics") or [])
        harmonized["document_role"] = normalize_document_role(
            harmonized.get("document_role"),
            page.get("source_type"),
            harmonized.get("brief_title"),
        )
        harmonized["primary_people"] = harmonize_entity_refs(
            harmonized.get("primary_people") or [],
            people,
            label_keys=("display_name", "alternate_names"),
            id_key="id",
            unresolved_prefix="person",
        )
        harmonized["primary_locations"] = harmonize_entity_refs(
            harmonized.get("primary_locations") or [],
            locations,
            label_keys=("name", "address_1886", "address_1887", "modern_address"),
            id_key="id",
            unresolved_prefix="location",
        )
        harmonized["speaker_directory"] = harmonize_speaker_directory(
            harmonized.get("speaker_directory") or [],
            people,
        )

        referenced, document_events, routine = classify_brief_events(
            source_id=source_id,
            referenced_events=harmonized.get("referenced_events") or [],
            document_events=harmonized.get("document_events") or [],
            people=people,
            locations=locations,
            events=events,
        )
        harmonized["referenced_events"] = referenced
        harmonized["document_events"] = document_events
        harmonized["routine_procedural_events"] = routine
        harmonized["harmonization"] = {
            "version": HARMONIZATION_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "raw_event_reference_count": len(raw_briefing.get("referenced_events") or []),
            "raw_document_event_count": len(raw_briefing.get("document_events") or []),
            "historical_event_count": len(referenced),
            "document_event_count": len(document_events),
            "routine_procedural_count": len(routine),
        }
        harmonized_by_source[source_id] = harmonized

        for ref in referenced:
            event_refs.append({**ref, "source_id": source_id})
            classification_counts[ref.get("event_class") or HISTORICAL_CLASS] += 1
        for ref in document_events:
            document_event_refs.append({**ref, "source_id": source_id})
            classification_counts[ref.get("event_class") or DOCUMENT_CLASS] += 1
        for ref in routine:
            routine_refs.append({**ref, "source_id": source_id})
            classification_counts[ROUTINE_CLASS] += 1

    report(
        "Brief harmonization: deterministic classification "
        f"{len(event_refs)} historical/uncertain, {len(document_event_refs)} document, {len(routine_refs)} routine"
    )
    classification_review_batches = review_ambiguous_classifications(
        storage=storage,
        raw_prefix=raw_prefix,
        event_refs=event_refs,
        document_event_refs=document_event_refs,
        routine_refs=routine_refs,
        enabled=use_llm_review,
        review_model=review_model,
        max_new_batches=max_llm_review_batches,
        report=report,
    )
    event_refs, document_event_refs, routine_refs = apply_classification_reviews(
        harmonized_by_source=harmonized_by_source,
        event_refs=event_refs,
        document_event_refs=document_event_refs,
        routine_refs=routine_refs,
        review_batches=classification_review_batches,
    )
    classification_counts = Counter()
    for ref in event_refs:
        classification_counts[ref.get("event_class") or HISTORICAL_CLASS] += 1
    for ref in document_event_refs:
        classification_counts[ref.get("event_class") or DOCUMENT_CLASS] += 1
    for _ref in routine_refs:
        classification_counts[ROUTINE_CLASS] += 1

    normalized_refs = build_normalized_refs(
        harmonized_by_source=harmonized_by_source,
        event_refs=event_refs,
        document_event_refs=document_event_refs,
        routine_refs=routine_refs,
    )
    write_intermediate_artifact(storage, raw_prefix, "normalized_refs.json", normalized_refs)
    report(f"Brief harmonization: normalized {len(normalized_refs)} reference(s)")
    embedding_inputs = build_embedding_inputs(
        normalized_refs=normalized_refs,
        people=people,
        locations=locations,
        events=events,
        harmonized_by_source=harmonized_by_source,
    )
    write_intermediate_artifact(storage, raw_prefix, "embedding_inputs.json", embedding_inputs)
    report(f"Brief harmonization: prepared {len(embedding_inputs)} embedding input(s)")
    vectors_by_id, embedding_cache_manifest = load_or_create_embeddings(
        storage=storage,
        embedding_inputs=embedding_inputs,
        model=embedding_model,
        dimensions=embedding_dimensions,
        enabled=use_embeddings,
        report=report,
    )
    write_intermediate_artifact(storage, raw_prefix, "embedding_cache_manifest.json", embedding_cache_manifest)
    report("Brief harmonization: building candidate matches")
    candidate_matches = build_candidate_matches(normalized_refs, vectors_by_id, embedding_inputs, report=report)
    write_intermediate_artifact(storage, raw_prefix, "candidate_matches.json", candidate_matches)
    report(
        "Brief harmonization: built candidate matches "
        f"{dict(Counter(match.get('decision') for match in candidate_matches))}"
    )
    remaining_review_cap = remaining_new_review_batches(max_llm_review_batches, classification_review_batches)
    llm_review_batches = classification_review_batches + review_ambiguous_candidates(
        storage=storage,
        raw_prefix=raw_prefix,
        candidate_matches=candidate_matches,
        normalized_refs=normalized_refs,
        enabled=use_llm_review,
        review_model=review_model,
        escalation_review_model=escalation_review_model,
        max_new_batches=remaining_review_cap,
        max_review_candidates=max_llm_review_candidates,
        report=report,
    )
    approved_pairs = approved_match_pairs(candidate_matches, llm_review_batches)

    event_clusters = cluster_event_refs(event_refs, "brief_event", is_document=False, approved_pairs=approved_pairs)
    document_event_clusters = cluster_event_refs(
        document_event_refs,
        "brief_document_event",
        is_document=True,
        approved_pairs=approved_pairs,
    )
    proposed_event_clusters = copy.deepcopy(event_clusters)
    proposed_document_event_clusters = copy.deepcopy(document_event_clusters)
    refs_by_id = {ref["ref_id"]: ref for ref in normalized_refs}
    event_cluster_second_pass = merge_aligned_event_clusters(
        event_clusters,
        vectors_by_id=vectors_by_id,
        refs_by_id=refs_by_id,
        is_document=False,
    )
    document_cluster_second_pass = merge_aligned_event_clusters(
        document_event_clusters,
        vectors_by_id=vectors_by_id,
        refs_by_id=refs_by_id,
        is_document=True,
    )
    report(
        "Brief harmonization: second-pass cluster merges "
        f"{event_cluster_second_pass['merged_clusters']} event cluster(s), "
        f"{document_cluster_second_pass['merged_clusters']} document-event cluster(s)"
    )
    write_intermediate_artifact(
        storage,
        raw_prefix,
        "cluster_second_pass_matches.json",
        {
            "event_clusters": event_cluster_second_pass,
            "document_event_clusters": document_cluster_second_pass,
        },
    )
    proposed_clusters = {
        "event_clusters": proposed_event_clusters,
        "document_event_clusters": proposed_document_event_clusters,
    }
    write_intermediate_artifact(storage, raw_prefix, "proposed_clusters.json", proposed_clusters)
    final_clusters = {
        "event_clusters": event_clusters,
        "document_event_clusters": document_event_clusters,
        "routine_procedural_refs": routine_refs,
    }
    write_intermediate_artifact(storage, raw_prefix, "final_clusters.json", final_clusters)
    assign_cluster_ids(harmonized_by_source, event_clusters, "referenced_events", "navigation_event_id")
    assign_cluster_ids(harmonized_by_source, document_event_clusters, "document_events", "navigation_document_event_id")

    coverage = build_coverage(
        pages=pages,
        briefings_by_source=briefings_by_source,
        harmonized_by_source=harmonized_by_source,
        event_clusters=event_clusters,
        document_event_clusters=document_event_clusters,
        routine_refs=routine_refs,
    )
    qa_report = build_qa_report(
        coverage=coverage,
        candidate_matches=candidate_matches,
        llm_review_batches=llm_review_batches,
        final_clusters=final_clusters,
    )
    write_intermediate_artifact(storage, raw_prefix, "qa_report.json", qa_report)
    artifact = {
        "run_id": run_id,
        "version": HARMONIZATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": harmonized_by_source,
        "event_clusters": event_clusters,
        "document_event_clusters": document_event_clusters,
        "routine_procedural_refs": routine_refs,
        "coverage": coverage,
        "classification_counts": dict(classification_counts),
        "methods": {
            "deterministic": True,
            "character_similarity": True,
            "embedding_cosine": {
                "enabled": use_embeddings,
                "model": embedding_model,
                "dimensions": embedding_dimensions,
                "cache_hits": embedding_cache_manifest.get("cache_hits", 0),
                "created": embedding_cache_manifest.get("created", 0),
            },
            "llm_confirmation": {
                "enabled": use_llm_review,
                "review_model": review_model,
                "review_reasoning_effort": DEFAULT_REVIEW_REASONING_EFFORT,
                "escalation_review_model": escalation_review_model,
                "escalation_reasoning_effort": DEFAULT_ESCALATION_REASONING_EFFORT,
                "batches": len(llm_review_batches),
                "max_new_batches": max_llm_review_batches,
                "max_review_candidates": max_llm_review_candidates,
            },
        },
    }
    write_intermediate_artifact(storage, raw_prefix, "audit.json", artifact)
    report(
        "Brief harmonization: complete "
        f"{len(event_clusters)} event cluster(s), {len(document_event_clusters)} document-event cluster(s), "
        f"coverage_ok={coverage.get('coverage_ok')}"
    )
    return {
        "briefings_by_source": harmonized_by_source,
        "artifact": artifact,
        "coverage": coverage,
        "event_clusters": event_clusters,
        "document_event_clusters": document_event_clusters,
        "normalized_refs": normalized_refs,
        "embedding_inputs": embedding_inputs,
        "embedding_cache_manifest": embedding_cache_manifest,
        "candidate_matches": candidate_matches,
        "proposed_clusters": proposed_clusters,
        "llm_review_batches": llm_review_batches,
        "final_clusters": final_clusters,
        "qa_report": qa_report,
    }


def build_normalized_refs(
    *,
    harmonized_by_source: dict[str, dict[str, Any]],
    event_refs: list[dict[str, Any]],
    document_event_refs: list[dict[str, Any]],
    routine_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for source_id, briefing in harmonized_by_source.items():
        source_title = clean_text(briefing.get("brief_title")) or source_id
        for index, person in enumerate(briefing.get("primary_people") or []):
            refs.append(
                normalized_ref(
                    ref_id=f"{source_id}:primary_people:{index}",
                    source_id=source_id,
                    ref_type="person",
                    label=person.get("label"),
                    canonical_id=person.get("canonical_id"),
                    description=person.get("role_or_relationship"),
                    source_title=source_title,
                    confidence=person.get("confidence"),
                )
            )
        for index, location in enumerate(briefing.get("primary_locations") or []):
            refs.append(
                normalized_ref(
                    ref_id=f"{source_id}:primary_locations:{index}",
                    source_id=source_id,
                    ref_type="place",
                    label=location.get("label"),
                    canonical_id=location.get("canonical_id"),
                    description=location.get("role_or_relationship"),
                    source_title=source_title,
                    confidence=location.get("confidence"),
                )
            )
        for index, topic in enumerate(briefing.get("topics") or []):
            refs.append(
                normalized_ref(
                    ref_id=f"{source_id}:topics:{index}",
                    source_id=source_id,
                    ref_type="topic",
                    label=topic,
                    canonical_id=None,
                    description=briefing.get("navigation_summary"),
                    source_title=source_title,
                    confidence=None,
                )
            )

    for ref_type, refs_in_type in (
        ("historical_event", event_refs),
        ("document_event", document_event_refs),
        ("routine_procedural", routine_refs),
    ):
        for ref in refs_in_type:
            source_id = ref.get("source_id")
            briefing = harmonized_by_source.get(source_id, {})
            refs.append(
                normalized_ref(
                    ref_id=ref.get("source_event_ref_id"),
                    source_id=source_id,
                    ref_type=ref_type,
                    label=ref.get("label"),
                    canonical_id=ref.get("canonical_id"),
                    description=ref.get("summary"),
                    source_title=briefing.get("brief_title"),
                    confidence=ref.get("confidence"),
                    event_time=ref.get("event_time"),
                    location_label=ref.get("location_label"),
                    location_id=ref.get("location_id"),
                    participant_labels=ref.get("participant_labels"),
                    participant_person_ids=ref.get("participant_person_ids"),
                    supporting_quote=ref.get("supporting_quote"),
                    page_refs=ref.get("page_refs"),
                    event_kind=ref.get("event_kind"),
                    event_class=ref.get("event_class"),
                )
            )
    return [ref for ref in refs if ref.get("ref_id") and ref.get("label")]


def progress_reporter(enabled: bool):
    def report(message: str) -> None:
        if enabled:
            print(message, flush=True)

    return report


def write_intermediate_artifact(
    storage: JsonStorage | None,
    raw_prefix: str,
    filename: str,
    data: Any,
) -> None:
    if storage is None:
        return
    storage.write_json(f"{raw_prefix}/{filename}", data)


def review_batch_path(raw_prefix: str, batch_id: str) -> str:
    return f"{raw_prefix}/llm_review_batches/{batch_id}.json"


def read_completed_review_batch(storage: JsonStorage | None, raw_prefix: str, batch_id: str) -> dict[str, Any] | None:
    if storage is None:
        return None
    path = review_batch_path(raw_prefix, batch_id)
    if not storage.exists(path):
        return None
    batch = storage.read_json(path)
    if batch.get("status") == "success" and batch.get("output"):
        return batch
    return None


def count_new_review_batches(review_batches: list[dict[str, Any]]) -> int:
    return sum(1 for batch in review_batches if not batch.get("cache", {}).get("reused"))


def remaining_new_review_batches(
    max_new_batches: int | None,
    completed_batches: list[dict[str, Any]],
) -> int | None:
    if max_new_batches is None or max_new_batches < 0:
        return None
    return max(0, max_new_batches - count_new_review_batches(completed_batches))


def normalized_ref(
    *,
    ref_id: str | None,
    source_id: str | None,
    ref_type: str,
    label: Any,
    canonical_id: Any,
    description: Any,
    source_title: Any,
    confidence: Any,
    event_time: dict[str, Any] | None = None,
    location_label: Any = None,
    location_id: Any = None,
    participant_labels: list[Any] | None = None,
    participant_person_ids: list[Any] | None = None,
    supporting_quote: Any = None,
    page_refs: list[Any] | None = None,
    event_kind: Any = None,
    event_class: Any = None,
) -> dict[str, Any]:
    text_parts = [
        clean_text(label),
        clean_text(description),
        clean_text(source_title),
        clean_text(location_label),
        " ".join(clean_text(item) for item in participant_labels or [] if clean_text(item)),
        clean_text(supporting_quote),
    ]
    event_time = event_time or {}
    ref = {
        "ref_id": ref_id,
        "source_id": source_id,
        "ref_type": ref_type,
        "label": clean_text(label),
        "normalized_label": normalize_key(label),
        "canonical_id": canonical_id,
        "description": clean_text(description),
        "source_title": clean_text(source_title),
        "confidence": confidence,
        "event_time": normalize_event_time(event_time) if event_time else {},
        "location_label": clean_text(location_label),
        "location_key": normalize_key(location_label),
        "location_id": location_id,
        "participant_labels": [clean_text(item) for item in participant_labels or [] if clean_text(item)],
        "participant_person_ids": sorted(set(item for item in participant_person_ids or [] if item)),
        "supporting_quote": shorten_quote(supporting_quote),
        "page_refs": normalize_page_refs(page_refs or []),
        "event_kind": clean_text(event_kind),
        "event_class": clean_text(event_class),
        "embedding_text": clean_text(" | ".join(part for part in text_parts if part)),
    }
    ref["embedding_text_hash"] = sha256_text(ref["embedding_text"])
    ref["embedding_id"] = f"ref_{sha256_text(ref['ref_id'] + '|' + ref['embedding_text_hash'])[:16]}"
    return ref


def build_embedding_inputs(
    *,
    normalized_refs: list[dict[str, Any]],
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    events: list[dict[str, Any]],
    harmonized_by_source: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, stable_id: str, text: str, source_ref_id: str | None = None) -> str | None:
        text = clean_text(text)
        if not text:
            return None
        text_hash = sha256_text(text)
        embedding_id = f"emb_{sha256_text(f'{kind}|{stable_id}|{text_hash}')[:18]}"
        if embedding_id in seen:
            return embedding_id
        seen.add(embedding_id)
        inputs.append(
            {
                "embedding_id": embedding_id,
                "kind": kind,
                "stable_id": stable_id,
                "source_ref_id": source_ref_id,
                "text": text,
                "text_sha256": text_hash,
            }
        )
        return embedding_id

    for ref in normalized_refs:
        embedding_id = add(ref["ref_type"], ref["ref_id"], ref.get("embedding_text") or ref.get("label"), ref["ref_id"])
        if embedding_id:
            ref["embedding_id"] = embedding_id

    for person in people:
        labels = entity_labels(person, ("display_name", "alternate_names"))
        text = " | ".join(
            clean_text(part)
            for part in [
                "person",
                ", ".join(labels),
                person.get("role") or person.get("occupation"),
                person.get("bio") or person.get("summary") or person.get("description"),
                " ".join(person.get("source_labels") or []),
            ]
            if clean_text(part)
        )
        add("canonical_person", person.get("id") or sha256_text(text)[:12], text)

    for location in locations:
        text = " | ".join(
            clean_text(part)
            for part in [
                "place",
                location.get("name"),
                location.get("address_1886"),
                location.get("address_1887"),
                location.get("modern_address"),
                location.get("type") or location.get("location_type"),
                location.get("summary") or location.get("description"),
            ]
            if clean_text(part)
        )
        add("canonical_place", location.get("id") or sha256_text(text)[:12], text)

    for event in events:
        text = " | ".join(
            clean_text(part)
            for part in [
                "event",
                event.get("title") or event.get("label"),
                event_date_text(event.get("time") or event.get("event_time")),
                event.get("location_label") or event.get("location_name") or event.get("location_id"),
                ", ".join(event.get("participant_labels") or event.get("participant_person_ids") or []),
                event.get("description") or event.get("summary"),
            ]
            if clean_text(part)
        )
        add("canonical_event", event.get("id") or sha256_text(text)[:12], text)

    topic_examples: dict[str, list[str]] = {}
    for briefing in harmonized_by_source.values():
        for topic in briefing.get("topics") or []:
            topic_examples.setdefault(topic, [])
            if len(topic_examples[topic]) < 5:
                topic_examples[topic].append(
                    clean_text(f"{briefing.get('brief_title')} {briefing.get('navigation_summary')}")
                )
    for topic, examples in topic_examples.items():
        add("topic_anchor", slugify(topic, "topic"), " | ".join(["topic", topic, *examples]))

    return inputs


def load_or_create_embeddings(
    *,
    storage: JsonStorage | None,
    embedding_inputs: list[dict[str, Any]],
    model: str,
    dimensions: int,
    enabled: bool,
    report=None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    report = report or (lambda _message: None)
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "version": HARMONIZATION_VERSION,
        "enabled": enabled,
        "model": model,
        "dimensions": dimensions,
        "generated_at": generated_at,
        "cache_hits": 0,
        "created": 0,
        "usage": zero_usage(),
        "cost_usd": 0.0,
        "items": [],
    }
    vectors_by_id: dict[str, list[float]] = {}
    if not enabled:
        report("Brief harmonization: embeddings disabled")
        return vectors_by_id, manifest
    if storage is None:
        raise RuntimeError("Brief harmonization embeddings require a storage backend for cache artifacts.")

    missing: list[dict[str, Any]] = []
    for item in embedding_inputs:
        cache_key = embedding_cache_key(item, model, dimensions)
        cache_path = embedding_cache_path(cache_key)
        manifest_item = {
            "embedding_id": item["embedding_id"],
            "kind": item["kind"],
            "stable_id": item["stable_id"],
            "source_ref_id": item.get("source_ref_id"),
            "text_sha256": item["text_sha256"],
            "cache_key": cache_key,
            "cache_path": cache_path,
            "status": "missing",
        }
        if storage.exists(cache_path):
            cached = storage.read_json(cache_path)
            vectors_by_id[item["embedding_id"]] = cached.get("embedding") or []
            manifest["cache_hits"] += 1
            manifest_item["status"] = "hit"
        else:
            missing.append(item)
        manifest["items"].append(manifest_item)

    report(
        "Brief harmonization: embedding cache "
        f"{manifest['cache_hits']} hit(s), {len(missing)} missing "
        f"({model}, dimensions={dimensions})"
    )
    for batch in chunks(missing, 96):
        batch_number = manifest["created"] // 96 + 1
        total_batches = math.ceil(len(missing) / 96) if missing else 0
        report(f"Brief harmonization: embedding batch {batch_number}/{total_batches} ({len(batch)} input(s))")
        vectors, raw_output, usage = call_openai_embeddings(
            model=model,
            inputs=[item["text"] for item in batch],
            dimensions=dimensions,
        )
        if len(vectors) != len(batch):
            raise RuntimeError(f"Embedding response count mismatch: expected {len(batch)}, received {len(vectors)}")
        add_usage(manifest["usage"], usage)
        batch_cost = estimate_cost_usd(model, usage)
        manifest["cost_usd"] += batch_cost
        for item, vector in zip(batch, vectors):
            cache_key = embedding_cache_key(item, model, dimensions)
            cache_path = embedding_cache_path(cache_key)
            vectors_by_id[item["embedding_id"]] = vector
            storage.write_json(
                cache_path,
                {
                    "version": HARMONIZATION_VERSION,
                    "model": model,
                    "dimensions": dimensions,
                    "input_text_hash": item["text_sha256"],
                    "embedding_id": item["embedding_id"],
                    "generated_at": generated_at,
                    "usage": usage,
                    "raw_model": (raw_output or {}).get("model"),
                    "embedding": vector,
                },
            )
            manifest["created"] += 1
            for manifest_item in manifest["items"]:
                if manifest_item["embedding_id"] == item["embedding_id"]:
                    manifest_item["status"] = "created"
                    break
        report(
            f"Brief harmonization: embedding batch {batch_number}/{total_batches} done "
            f"cost=${batch_cost:.6f}"
        )
    manifest["cost_usd"] = round(manifest["cost_usd"], 8)
    return vectors_by_id, manifest


def build_candidate_matches(
    normalized_refs: list[dict[str, Any]],
    vectors_by_id: dict[str, list[float]],
    embedding_inputs: list[dict[str, Any]] | None = None,
    report=None,
) -> list[dict[str, Any]]:
    report = report or (lambda _message: None)
    refs_by_type: dict[str, list[dict[str, Any]]] = {}
    for ref in normalized_refs:
        refs_by_type.setdefault(candidate_group(ref["ref_type"]), []).append(ref)

    matches: list[dict[str, Any]] = []
    for group, refs in refs_by_type.items():
        pair_estimate = len(refs) * max(len(refs) - 1, 0) // 2
        report(f"Brief harmonization: candidate group {group}: {len(refs)} ref(s), up to {pair_estimate} pair(s)")
        considered = 0
        scored = 0
        for left_index, left in enumerate(refs):
            if left_index and left_index % 500 == 0:
                report(
                    f"Brief harmonization: candidate group {group}: "
                    f"{left_index}/{len(refs)} refs scanned, {scored} candidate(s)"
                )
            for right in refs[left_index + 1 :]:
                if left.get("source_id") == right.get("source_id") and group != "event":
                    continue
                considered += 1
                if not candidate_prefilter(left, right, group):
                    continue
                match = score_candidate_pair(left, right, group, vectors_by_id)
                if match:
                    matches.append(match)
                    scored += 1
        report(
            f"Brief harmonization: candidate group {group}: "
            f"{considered} pair(s) considered, {scored} candidate(s)"
        )
    for anchor in canonical_anchor_refs(embedding_inputs or []):
        group = candidate_group(anchor["ref_type"])
        anchor_scored = 0
        for ref in refs_by_type.get(group, []):
            if not candidate_prefilter(ref, anchor, group):
                continue
            match = score_candidate_pair(ref, anchor, group, vectors_by_id)
            if match:
                match["right_is_canonical_anchor"] = True
                matches.append(match)
                anchor_scored += 1
        if anchor_scored:
            report(f"Brief harmonization: canonical anchor {anchor['ref_id']} produced {anchor_scored} candidate(s)")
    return sorted(matches, key=lambda item: (item["decision"], -float(item["overall_score"]), item["pair_id"]))


def candidate_prefilter(left: dict[str, Any], right: dict[str, Any], group: str) -> bool:
    if left.get("canonical_id") and left.get("canonical_id") == right.get("canonical_id"):
        return True
    left_label = left.get("normalized_label") or normalize_key(left.get("label"))
    right_label = right.get("normalized_label") or normalize_key(right.get("label"))
    if not left_label or not right_label:
        return False
    if left_label == right_label:
        return True
    left_tokens = significant_tokens(left_label)
    right_tokens = significant_tokens(right_label)
    shared = left_tokens.intersection(right_tokens)
    if group == "event":
        if not dates_compatible(left, right):
            return False
        if not shared and not (has_high_impact_anchor(left) and has_high_impact_anchor(right)):
            return False
        return True
    if group in {"person", "place"}:
        return bool(shared) or has_high_impact_anchor(left) or has_high_impact_anchor(right)
    return bool(shared)


def significant_tokens(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "by",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "source",
        "testimony",
        "document",
        "event",
    }
    return {token for token in normalize_key(value).split() if len(token) > 2 and token not in stopwords}


def distinctive_location_tokens(value: Any) -> set[str]:
    generic = {
        "ave",
        "avenue",
        "chicago",
        "city",
        "county",
        "court",
        "des",
        "illinois",
        "near",
        "place",
        "plaines",
        "square",
        "st",
        "street",
        "the",
        "west",
    }
    return {token for token in significant_tokens(normalize_key(value)) if token not in generic}


def canonical_anchor_refs(embedding_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    kind_to_type = {
        "canonical_person": "person",
        "canonical_place": "place",
        "canonical_event": "historical_event",
        "topic_anchor": "topic",
    }
    for item in embedding_inputs:
        ref_type = kind_to_type.get(item.get("kind"))
        if not ref_type:
            continue
        text = clean_text(item.get("text"))
        refs.append(
            {
                "ref_id": f"anchor:{item.get('stable_id')}",
                "source_id": None,
                "ref_type": ref_type,
                "label": text[:160],
                "normalized_label": normalize_key(text),
                "canonical_id": item.get("stable_id") if item.get("kind", "").startswith("canonical_") else None,
                "description": text,
                "source_title": "canonical anchor",
                "event_time": {},
                "location_label": "",
                "location_id": None,
                "participant_labels": [],
                "participant_person_ids": [],
                "embedding_id": item.get("embedding_id"),
            }
        )
    return refs


def score_candidate_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    group: str,
    vectors_by_id: dict[str, list[float]],
) -> dict[str, Any] | None:
    left_label = left.get("normalized_label") or ""
    right_label = right.get("normalized_label") or ""
    if not left_label or not right_label:
        return None
    exact = left_label == right_label
    char_score = SequenceMatcher(None, left_label, right_label).ratio()
    token_score = token_overlap(left_label, right_label)
    local_score = cosine_similarity(left_label, right_label)
    embedding_score = vector_cosine(vectors_by_id.get(left.get("embedding_id")), vectors_by_id.get(right.get("embedding_id")))
    semantic_score = embedding_score if embedding_score is not None else local_score
    date_compatible = dates_compatible(left, right)
    place_compatible = places_compatible(left, right)
    participant_score = participant_overlap(left, right)
    canonical_match = bool(left.get("canonical_id") and left.get("canonical_id") == right.get("canonical_id"))
    context_ok = date_compatible and place_compatible
    overall = max(char_score, token_score, local_score, semantic_score or 0.0)
    if group == "event":
        overall = max(overall, (semantic_score or 0.0) * 0.65 + participant_score * 0.2 + (0.15 if context_ok else 0.0))

    high_impact = has_high_impact_anchor(left) or has_high_impact_anchor(right)
    decision = "rejected"
    if canonical_match:
        decision = "accepted_auto"
    elif exact and context_ok:
        decision = "accepted_auto"
    elif group == "event" and context_ok and (semantic_score or 0.0) >= 0.88 and participant_score >= 0.1:
        decision = "accepted_auto"
    elif group == "event" and context_ok and (semantic_score or 0.0) >= 0.78:
        decision = "needs_llm_review"
    elif group != "event" and context_ok and overall >= 0.92:
        decision = "accepted_auto"
    elif context_ok and overall >= 0.82:
        decision = "needs_llm_review"
    elif high_impact and overall >= 0.70:
        decision = "needs_llm_review"

    if decision == "rejected" and overall < 0.72:
        return None
    pair_id = pair_id_for(left["ref_id"], right["ref_id"])
    return {
        "pair_id": pair_id,
        "group": group,
        "left_ref_id": left["ref_id"],
        "right_ref_id": right["ref_id"],
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "left_type": left.get("ref_type"),
        "right_type": right.get("ref_type"),
        "scores": {
            "exact_label": 1.0 if exact else 0.0,
            "character_similarity": round(char_score, 4),
            "token_overlap": round(token_score, 4),
            "local_char_ngram_cosine": round(local_score, 4),
            "embedding_cosine": round(embedding_score, 4) if embedding_score is not None else None,
            "participant_overlap": round(participant_score, 4),
        },
        "overall_score": round(overall, 4),
        "constraints": {
            "date_compatible": date_compatible,
            "place_compatible": place_compatible,
            "canonical_match": canonical_match,
            "high_impact_anchor": high_impact,
        },
        "decision": decision,
    }


def review_ambiguous_classifications(
    *,
    storage: JsonStorage | None,
    raw_prefix: str,
    event_refs: list[dict[str, Any]],
    document_event_refs: list[dict[str, Any]],
    routine_refs: list[dict[str, Any]],
    enabled: bool,
    review_model: str,
    max_new_batches: int | None,
    report=None,
) -> list[dict[str, Any]]:
    report = report or (lambda _message: None)
    del routine_refs
    candidates = [
        ref
        for ref in [*event_refs, *document_event_refs]
        if ref.get("event_class") in {UNCERTAIN_CLASS, BOTH_CLASS}
    ]
    if not enabled or not candidates:
        if candidates:
            report(f"Brief harmonization: skipped {len(candidates)} ambiguous classification ref(s); LLM review disabled")
        else:
            report("Brief harmonization: no ambiguous classification refs for LLM review")
        return []
    batches: list[dict[str, Any]] = []
    candidate_batches = chunks(candidates, 32)
    report(
        "Brief harmonization: classification LLM review "
        f"{len(candidates)} ref(s), {len(candidate_batches)} batch(es), "
        f"new-batch cap={max_new_batches if max_new_batches is not None and max_new_batches >= 0 else 'none'}"
    )
    new_batches_called = 0
    for index, ref_batch in enumerate(candidate_batches, start=1):
        batch_id = f"classification_review_batch_{index:04d}"
        cached = read_completed_review_batch(storage, raw_prefix, batch_id)
        if cached:
            cached["cache"] = {"reused": True}
            report(f"Brief harmonization: classification batch {index}/{len(candidate_batches)} cache hit")
            batches.append(cached)
            continue
        llm_input = {
            "batch_id": batch_id,
            "instructions": (
                "Classify each archival event reference as historical_event, document_event, both, "
                "routine_procedural, or uncertain. Preserve routine procedural items in audit."
            ),
            "refs": [compact_ref_for_review(normalized_ref_for_classification(ref)) for ref in ref_batch],
        }
        record = {
            "batch_id": batch_id,
            "review_type": "event_classification",
            "model": review_model,
            "reasoning_effort": DEFAULT_REVIEW_REASONING_EFFORT,
            "high_risk": False,
            "input": llm_input,
            "status": "pending",
            "output": None,
            "raw_output": None,
            "usage": zero_usage(),
            "cost_usd": 0.0,
        }
        if max_new_batches is not None and max_new_batches >= 0 and new_batches_called >= max_new_batches:
            record["status"] = "skipped_cap"
            record["error"] = "LLM review batch cap reached; rerun with a higher cap to continue."
            write_intermediate_artifact(storage, raw_prefix, f"llm_review_batches/{batch_id}.json", record)
            report(f"Brief harmonization: classification batch {index}/{len(candidate_batches)} skipped by cap")
            batches.append(record)
            continue
        write_intermediate_artifact(storage, raw_prefix, f"llm_review_batches/{batch_id}.json", record)
        new_batches_called += 1
        report(f"Brief harmonization: classification batch {index}/{len(candidate_batches)} starting ({review_model})")
        try:
            parsed, raw_output, usage = call_openai_structured(
                model=review_model,
                input_messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious archival event classifier. "
                            "Separate historical events from document lifecycle and routine court-procedure events."
                        ),
                    },
                    {"role": "user", "content": json_dumps(llm_input)},
                ],
                schema=classification_review_schema(),
                schema_name="brief_event_classification_review",
                max_output_tokens=5000,
                reasoning_effort=DEFAULT_REVIEW_REASONING_EFFORT,
            )
            record["status"] = "success"
            record["output"] = parsed
            record["raw_output"] = raw_output
            record["usage"] = usage
            record["cost_usd"] = round(estimate_cost_usd(review_model, usage), 8)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        write_intermediate_artifact(storage, raw_prefix, f"llm_review_batches/{batch_id}.json", record)
        report(
            "Brief harmonization: classification batch "
            f"{index}/{len(candidate_batches)} {record['status']} "
            f"cost=${float(record.get('cost_usd') or 0.0):.6f}"
        )
        batches.append(record)
    return batches


def normalized_ref_for_classification(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref_id": ref.get("source_event_ref_id"),
        "ref_type": ref.get("event_class") or UNCERTAIN_CLASS,
        "label": ref.get("label"),
        "canonical_id": ref.get("canonical_id"),
        "event_time": ref.get("event_time") or {},
        "location_label": ref.get("location_label"),
        "participant_labels": ref.get("participant_labels") or [],
        "participant_person_ids": ref.get("participant_person_ids") or [],
        "description": ref.get("summary"),
        "supporting_quote": ref.get("supporting_quote"),
        "source_title": ref.get("source_id"),
    }


def classification_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ref_id", "event_class", "confidence", "rationale"],
                    "properties": {
                        "ref_id": {"type": "string"},
                        "event_class": {
                            "type": "string",
                            "enum": [HISTORICAL_CLASS, DOCUMENT_CLASS, BOTH_CLASS, ROUTINE_CLASS, UNCERTAIN_CLASS],
                        },
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


def apply_classification_reviews(
    *,
    harmonized_by_source: dict[str, dict[str, Any]],
    event_refs: list[dict[str, Any]],
    document_event_refs: list[dict[str, Any]],
    routine_refs: list[dict[str, Any]],
    review_batches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: dict[str, str] = {}
    for batch in review_batches:
        if batch.get("review_type") != "event_classification" or batch.get("status") != "success":
            continue
        for decision in (batch.get("output") or {}).get("decisions") or []:
            if float(decision.get("confidence") or 0.0) >= 0.6:
                decisions[decision.get("ref_id")] = decision.get("event_class")
    if not decisions:
        return event_refs, document_event_refs, routine_refs

    refs_by_id: dict[str, dict[str, Any]] = {}
    for ref in [*event_refs, *document_event_refs, *routine_refs]:
        ref_id = ref.get("source_event_ref_id")
        if ref_id and ref_id not in refs_by_id:
            refs_by_id[ref_id] = ref
    new_event_refs: list[dict[str, Any]] = []
    new_document_refs: list[dict[str, Any]] = []
    new_routine_refs: list[dict[str, Any]] = []
    for ref_id, ref in refs_by_id.items():
        event_class = decisions.get(ref_id) or ref.get("event_class") or UNCERTAIN_CLASS
        if event_class == ROUTINE_CLASS:
            updated = {**ref, "event_class": ROUTINE_CLASS}
            new_routine_refs.append(updated)
        elif event_class == DOCUMENT_CLASS:
            updated = {**ref, "event_class": DOCUMENT_CLASS, "event_kind": ref.get("event_kind") or infer_document_event_kind(ref)}
            new_document_refs.append(updated)
        elif event_class == BOTH_CLASS:
            historical = {**ref, "event_class": BOTH_CLASS}
            document = {**ref, "event_class": BOTH_CLASS, "event_kind": ref.get("event_kind") or infer_document_event_kind(ref)}
            new_event_refs.append(historical)
            new_document_refs.append(document)
        else:
            updated = {**ref, "event_class": event_class}
            new_event_refs.append(updated)

    for briefing in harmonized_by_source.values():
        briefing["referenced_events"] = []
        briefing["document_events"] = []
        briefing["routine_procedural_events"] = []
    for field, refs in (
        ("referenced_events", new_event_refs),
        ("document_events", new_document_refs),
        ("routine_procedural_events", new_routine_refs),
    ):
        for ref in refs:
            source_id = ref.get("source_id")
            if source_id not in harmonized_by_source:
                continue
            clean_ref = {key: value for key, value in ref.items() if key != "source_id"}
            harmonized_by_source[source_id].setdefault(field, []).append(clean_ref)
    return new_event_refs, new_document_refs, new_routine_refs


def review_ambiguous_candidates(
    *,
    storage: JsonStorage | None,
    raw_prefix: str,
    candidate_matches: list[dict[str, Any]],
    normalized_refs: list[dict[str, Any]],
    enabled: bool,
    review_model: str,
    escalation_review_model: str,
    max_new_batches: int | None,
    max_review_candidates: int | None,
    report=None,
) -> list[dict[str, Any]]:
    report = report or (lambda _message: None)
    ambiguous_raw = [match for match in candidate_matches if match.get("decision") == "needs_llm_review"]
    reviewable: list[dict[str, Any]] = []
    skipped_nonreviewable = 0
    for match in ambiguous_raw:
        if llm_reviewable_match(match):
            match["review_status"] = "candidate"
            reviewable.append(match)
        else:
            match["review_status"] = "not_reviewed_nonreviewable_group"
            skipped_nonreviewable += 1
    reviewable.sort(key=llm_review_priority)
    if max_review_candidates is not None and max_review_candidates >= 0:
        ambiguous = reviewable[:max_review_candidates]
        for match in ambiguous:
            match["review_status"] = "selected_for_llm_review"
        for match in reviewable[max_review_candidates:]:
            match["review_status"] = "not_reviewed_candidate_cap"
    else:
        ambiguous = reviewable
        for match in ambiguous:
            match["review_status"] = "selected_for_llm_review"
    if not enabled or not ambiguous:
        if ambiguous_raw:
            report(
                "Brief harmonization: skipped merge LLM review "
                f"({len(ambiguous_raw)} ambiguous candidate(s), {len(reviewable)} reviewable); "
                "LLM review disabled"
            )
        else:
            report("Brief harmonization: no ambiguous merge candidates for LLM review")
        return []

    refs_by_id = {ref["ref_id"]: ref for ref in normalized_refs}
    batches: list[dict[str, Any]] = []
    match_batches = chunks(ambiguous, 24)
    report(
        "Brief harmonization: merge LLM review "
        f"{len(ambiguous_raw)} ambiguous candidate(s), {len(reviewable)} reviewable, "
        f"{skipped_nonreviewable} skipped as non-reviewable, {len(ambiguous)} selected, "
        f"{len(match_batches)} batch(es), "
        f"new-batch cap={max_new_batches if max_new_batches is not None and max_new_batches >= 0 else 'none'}, "
        f"candidate cap={max_review_candidates if max_review_candidates is not None and max_review_candidates >= 0 else 'none'}"
    )
    new_batches_called = 0
    for index, match_batch in enumerate(match_batches, start=1):
        high_risk = any(is_high_risk_match(match) for match in match_batch)
        model = escalation_review_model if high_risk else review_model
        reasoning_effort = DEFAULT_ESCALATION_REASONING_EFFORT if high_risk else DEFAULT_REVIEW_REASONING_EFFORT
        batch_id = f"review_batch_{index:04d}"
        cached = read_completed_review_batch(storage, raw_prefix, batch_id)
        if cached:
            cached["cache"] = {"reused": True}
            report(f"Brief harmonization: merge batch {index}/{len(match_batches)} cache hit")
            batches.append(cached)
            continue
        llm_input = {
            "batch_id": batch_id,
            "instructions": (
                "Review only these candidate pairs. Decide whether each pair should merge, stay separate, "
                "or be sent to human review. Do not invent new clusters."
            ),
            "candidates": [
                {
                    "pair_id": match["pair_id"],
                    "scores": match["scores"],
                    "constraints": match["constraints"],
                    "left": compact_ref_for_review(refs_by_id.get(match["left_ref_id"], {})),
                    "right": compact_ref_for_review(refs_by_id.get(match["right_ref_id"], {})),
                }
                for match in match_batch
            ],
        }
        record = {
            "batch_id": batch_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "high_risk": high_risk,
            "input": llm_input,
            "status": "pending",
            "output": None,
            "raw_output": None,
            "usage": zero_usage(),
            "cost_usd": 0.0,
        }
        if max_new_batches is not None and max_new_batches >= 0 and new_batches_called >= max_new_batches:
            record["status"] = "skipped_cap"
            record["error"] = "LLM review batch cap reached; rerun with a higher cap to continue."
            write_intermediate_artifact(storage, raw_prefix, f"llm_review_batches/{batch_id}.json", record)
            report(f"Brief harmonization: merge batch {index}/{len(match_batches)} skipped by cap")
            batches.append(record)
            continue
        write_intermediate_artifact(storage, raw_prefix, f"llm_review_batches/{batch_id}.json", record)
        new_batches_called += 1
        report(f"Brief harmonization: merge batch {index}/{len(match_batches)} starting ({model})")
        try:
            parsed, raw_output, usage = call_openai_structured(
                model=model,
                input_messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious archival-data harmonization reviewer. "
                            "Prefer keeping records separate when evidence conflicts."
                        ),
                    },
                    {"role": "user", "content": json_dumps(llm_input)},
                ],
                schema=llm_review_schema(),
                schema_name="brief_harmonization_review",
                max_output_tokens=6000,
                reasoning_effort=reasoning_effort,
            )
            record["status"] = "success"
            record["output"] = parsed
            record["raw_output"] = raw_output
            record["usage"] = usage
            record["cost_usd"] = round(estimate_cost_usd(model, usage), 8)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        write_intermediate_artifact(storage, raw_prefix, f"llm_review_batches/{batch_id}.json", record)
        report(
            "Brief harmonization: merge batch "
            f"{index}/{len(match_batches)} {record['status']} "
            f"cost=${float(record.get('cost_usd') or 0.0):.6f}"
        )
        batches.append(record)
    return batches


def llm_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pair_id", "decision", "confidence", "rationale"],
                    "properties": {
                        "pair_id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": ["merge", "keep_separate", "needs_human_review"],
                        },
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


def approved_match_pairs(candidate_matches: list[dict[str, Any]], llm_review_batches: list[dict[str, Any]]) -> set[frozenset[str]]:
    approved: set[frozenset[str]] = set()
    by_pair_id = {match["pair_id"]: match for match in candidate_matches}
    for match in candidate_matches:
        if match.get("decision") == "accepted_auto":
            approved.add(frozenset((match["left_ref_id"], match["right_ref_id"])))
    for batch in llm_review_batches:
        output = batch.get("output") or {}
        for decision in output.get("decisions") or []:
            if decision.get("decision") != "merge":
                continue
            match = by_pair_id.get(decision.get("pair_id"))
            if match:
                approved.add(frozenset((match["left_ref_id"], match["right_ref_id"])))
    return approved


def build_qa_report(
    *,
    coverage: dict[str, Any],
    candidate_matches: list[dict[str, Any]],
    llm_review_batches: list[dict[str, Any]],
    final_clusters: dict[str, Any],
) -> dict[str, Any]:
    cluster_ref_ids: list[str] = []
    for cluster_group in ("event_clusters", "document_event_clusters"):
        for cluster in final_clusters.get(cluster_group) or []:
            cluster_ref_ids.extend(
                support.get("source_event_ref_id")
                for support in cluster.get("supporting_sources") or []
                if support.get("source_event_ref_id")
            )
    cluster_ref_ids.extend(
        ref.get("source_event_ref_id")
        for ref in final_clusters.get("routine_procedural_refs") or []
        if ref.get("source_event_ref_id")
    )
    duplicates = sorted(ref_id for ref_id, count in Counter(cluster_ref_ids).items() if count > 1)
    return {
        "version": HARMONIZATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage_ok": coverage.get("coverage_ok") is True and not duplicates,
        "coverage": coverage,
        "duplicate_final_ref_ids": duplicates,
        "candidate_match_counts": dict(Counter(match.get("decision") for match in candidate_matches)),
        "candidate_review_status_counts": dict(Counter(match.get("review_status", "not_applicable") for match in candidate_matches)),
        "llm_review": {
            "batches": len(llm_review_batches),
            "successful_batches": sum(1 for batch in llm_review_batches if batch.get("status") == "success"),
            "errored_batches": sum(1 for batch in llm_review_batches if batch.get("status") == "error"),
            "cost_usd": round(sum(float(batch.get("cost_usd") or 0.0) for batch in llm_review_batches), 8),
        },
        "qa_llm": {
            "enabled": False,
            "cost_usd": 0.0,
            "note": "QA report is deterministic; LLM costs here are review-batch costs, not a separate QA call.",
        },
        "risky_merges": [
            match
            for match in candidate_matches
            if match.get("decision") == "accepted_auto"
            and match.get("constraints", {}).get("high_impact_anchor")
            and float(match.get("overall_score") or 0.0) < 0.9
        ][:50],
        "over_split_anchor_candidates": [
            match for match in candidate_matches if match.get("decision") == "needs_llm_review"
        ][:50],
    }


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_topics(topics: list[Any]) -> list[str]:
    normalized: set[str] = set()
    for topic in topics:
        label = normalize_key(topic)
        if not label:
            continue
        for needle, replacement in TOPIC_SYNONYMS.items():
            if needle in label:
                label = replacement
                break
        normalized.add(label)
    return sorted(normalized)


def normalize_document_role(value: Any, source_type: Any, title: Any) -> str:
    role = normalize_key(value)
    allowed = {"testimony", "exhibit", "procedural", "toc", "cover", "legal_document", "other"}
    if role in allowed:
        return role
    source_role = normalize_key(source_type)
    if source_role in allowed:
        return source_role
    title_key = normalize_key(title)
    if "cover" in title_key:
        return "cover"
    if any(term in title_key for term in ("motion", "petition", "judgment", "sentence", "affidavit")):
        return "legal_document"
    return "other"


def harmonize_entity_refs(
    refs: list[Any],
    entities: list[dict[str, Any]],
    *,
    label_keys: tuple[str, ...],
    id_key: str,
    unresolved_prefix: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        label = clean_text(ref.get("label"))
        if not label and not ref.get("canonical_id"):
            continue
        match, method, score = find_entity_match(ref.get("canonical_id"), label, entities, id_key, label_keys)
        results.append(
            {
                "label": entity_label(match, label_keys) if match else label,
                "canonical_id": match.get(id_key) if match else ref.get("canonical_id"),
                "role_or_relationship": ref.get("role_or_relationship"),
                "confidence": ref.get("confidence"),
                "harmonized_key": (match or {}).get(id_key) or slugify(label, unresolved_prefix),
                "match_method": method,
                "match_score": score,
            }
        )
    return dedupe_refs(results, ("canonical_id", "label", "role_or_relationship"))


def harmonize_speaker_directory(directory: list[Any], people: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in directory:
        if not isinstance(entry, dict):
            continue
        display_name = clean_text(entry.get("display_name"))
        if not display_name:
            continue
        match, method, score = find_entity_match(
            entry.get("speaker_id"),
            display_name,
            people,
            "id",
            ("display_name", "alternate_names"),
        )
        speaker_id = match.get("id") if match else slugify(display_name, "person")
        if speaker_id in seen:
            continue
        seen.add(speaker_id)
        results.append(
            {
                "speaker_id": speaker_id,
                "display_name": entity_label(match, ("display_name",)) if match else display_name,
                "role": entry.get("role"),
                "match_method": method,
                "match_score": score,
            }
        )
    return results


def find_entity_match(
    canonical_id: Any,
    label: str,
    entities: list[dict[str, Any]],
    id_key: str,
    label_keys: tuple[str, ...],
) -> tuple[dict[str, Any] | None, str, float | None]:
    if canonical_id:
        match = next((entity for entity in entities if entity.get(id_key) == canonical_id), None)
        if match:
            return match, "canonical_id", 1.0
    label_key = normalize_key(label)
    if not label_key:
        return None, "unresolved", None
    best: tuple[dict[str, Any] | None, str, float] = (None, "unresolved", 0.0)
    for entity in entities:
        for candidate in entity_labels(entity, label_keys):
            candidate_key = normalize_key(candidate)
            if not candidate_key:
                continue
            if candidate_key == label_key:
                return entity, "exact_label", 1.0
            char_score = SequenceMatcher(None, label_key, candidate_key).ratio()
            cosine_score = cosine_similarity(label_key, candidate_key)
            score = max(char_score, cosine_score)
            if score > best[2]:
                best = (entity, "fuzzy_label" if char_score >= cosine_score else "embedding_cosine", score)
    if best[0] and best[2] >= 0.92:
        return best
    return None, "unresolved", round(best[2], 3) if best[2] else None


def entity_labels(entity: dict[str, Any], label_keys: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for key in label_keys:
        value = entity.get(key)
        if isinstance(value, list):
            labels.extend(str(item) for item in value if item)
        elif value:
            labels.append(str(value))
    return labels


def entity_label(entity: dict[str, Any] | None, label_keys: tuple[str, ...]) -> str | None:
    if not entity:
        return None
    return next((label for label in entity_labels(entity, label_keys) if label), None)


def classify_brief_events(
    *,
    source_id: str,
    referenced_events: list[Any],
    document_events: list[Any],
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    historical: list[dict[str, Any]] = []
    document: list[dict[str, Any]] = []
    routine: list[dict[str, Any]] = []

    for index, ref in enumerate(referenced_events):
        if not isinstance(ref, dict):
            continue
        normalized = normalize_event_ref(ref, people, locations, events)
        normalized["source_event_ref_id"] = f"{source_id}:referenced_events:{index}"
        classification = classify_event_ref(normalized, source_field="referenced_events")
        normalized["event_class"] = classification
        if classification == DOCUMENT_CLASS:
            normalized["event_kind"] = normalized.get("event_kind") or infer_document_event_kind(normalized)
            document.append(normalized)
        elif classification == ROUTINE_CLASS:
            routine.append(normalized)
        else:
            historical.append(normalized)

    for index, ref in enumerate(document_events):
        if not isinstance(ref, dict):
            continue
        normalized = normalize_event_ref(ref, people, locations, events)
        normalized["source_event_ref_id"] = f"{source_id}:document_events:{index}"
        normalized["event_kind"] = normalized.get("event_kind") or infer_document_event_kind(normalized)
        classification = classify_event_ref(normalized, source_field="document_events")
        normalized["event_class"] = classification
        if classification in {HISTORICAL_CLASS, BOTH_CLASS}:
            historical_ref = copy.deepcopy(normalized)
            historical_ref.pop("event_kind", None)
            historical.append(historical_ref)
            if classification == BOTH_CLASS:
                document.append(normalized)
        elif classification == ROUTINE_CLASS:
            routine.append(normalized)
        else:
            document.append(normalized)
    return historical, document, routine


def normalize_event_ref(
    ref: dict[str, Any],
    people: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    label = clean_text(ref.get("label"))
    event = find_event(ref.get("canonical_id"), label, events)
    location, location_method, location_score = find_entity_match(
        ref.get("location_id"),
        clean_text(ref.get("location_label")),
        locations,
        "id",
        ("name", "address_1886", "address_1887", "modern_address"),
    )
    participant_ids = normalize_participants(ref, people)
    event_time = normalize_event_time(ref.get("event_time") or {})
    normalized = {
        "label": event.get("title") if event else label,
        "canonical_id": event.get("id") if event else ref.get("canonical_id"),
        "event_time": event_time,
        "location_label": entity_label(location, ("name",)) if location else clean_text(ref.get("location_label")) or None,
        "location_id": location.get("id") if location else ref.get("location_id"),
        "participant_labels": [clean_text(label) for label in ref.get("participant_labels") or [] if clean_text(label)],
        "participant_person_ids": participant_ids,
        "summary": clean_text(ref.get("summary")),
        "supporting_quote": shorten_quote(ref.get("supporting_quote")),
        "page_refs": normalize_page_refs(ref.get("page_refs") or []),
        "confidence": ref.get("confidence"),
        "event_kind": ref.get("event_kind"),
        "match_method": "canonical_event_id" if event else None,
        "location_match_method": location_method,
        "location_match_score": location_score,
    }
    if event:
        normalized["event_time"] = normalize_event_time(event.get("time") or event_time)
        normalized["summary"] = normalized["summary"] or clean_text(event.get("description"))
    return normalized


def find_event(canonical_id: Any, label: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if canonical_id:
        match = next((event for event in events if event.get("id") == canonical_id), None)
        if match:
            return match
    label_key = normalize_key(label)
    if not label_key:
        return None
    for event in events:
        if normalize_key(event.get("title")) == label_key:
            return event
    return None


def normalize_participants(ref: dict[str, Any], people: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for person_id in ref.get("participant_person_ids") or []:
        if isinstance(person_id, str) and person_id and person_id not in ids:
            ids.append(person_id)
    for label in ref.get("participant_labels") or []:
        match, _, _ = find_entity_match(None, clean_text(label), people, "id", ("display_name", "alternate_names"))
        person_id = match.get("id") if match else None
        if person_id and person_id not in ids:
            ids.append(person_id)
    return ids


def normalize_event_time(value: dict[str, Any]) -> dict[str, Any]:
    start = value.get("start") or value.get("normalized_date")
    return {
        "start": start,
        "end": value.get("end"),
        "normalized_date": str(start)[:10] if start else None,
        "precision": value.get("precision") or "unknown",
        "original_text": value.get("original_text") or value.get("display"),
    }


def classify_event_ref(ref: dict[str, Any], *, source_field: str) -> str:
    text = normalize_key(" ".join(str(ref.get(key) or "") for key in ("label", "summary", "event_kind")))
    if any(term in text for term in ROUTINE_TERMS):
        return ROUTINE_CLASS
    historical = any(term in text for term in HISTORICAL_TERMS)
    document = any(term in text for term in DOCUMENT_TERMS) or source_field == "document_events"
    if historical and document:
        return BOTH_CLASS
    if historical:
        return HISTORICAL_CLASS
    if document:
        return DOCUMENT_CLASS
    return UNCERTAIN_CLASS if source_field == "referenced_events" else DOCUMENT_CLASS


def infer_document_event_kind(ref: dict[str, Any]) -> str:
    text = normalize_key(" ".join(str(ref.get(key) or "") for key in ("label", "summary", "event_kind")))
    if any(term in text for term in ("publication", "published", "newspaper")):
        return "publication"
    if any(term in text for term in ("filing", "filed", "certification")):
        return "filing"
    if any(term in text for term in ("introduced", "introduction", "exhibit", "evidence")):
        return "evidence_introduction"
    if any(term in text for term in ("catalog", "source", "transcript", "cover", "index")):
        return "source_description"
    if any(term in text for term in ("petition", "motion", "order", "arraignment", "verdict", "sentence", "judgment", "writ", "court")):
        return "court_procedure"
    if any(term in text for term in ("created", "creation", "drawn", "made")):
        return "document_creation"
    return "other_document_event"


def cluster_event_refs(
    refs: list[dict[str, Any]],
    prefix: str,
    *,
    is_document: bool,
    approved_pairs: set[frozenset[str]] | None = None,
) -> list[dict[str, Any]]:
    approved_pairs = approved_pairs or set()
    clusters: list[dict[str, Any]] = []
    for ref in refs:
        key = conservative_event_key(ref, is_document=is_document)
        match = find_cluster(ref, key, clusters, is_document=is_document, approved_pairs=approved_pairs)
        if match is None:
            cluster_id = build_cluster_id(prefix, ref, len(clusters) + 1)
            match = {
                "id": cluster_id,
                "canonical_id": ref.get("canonical_id"),
                "label": ref.get("label"),
                "event_time": ref.get("event_time"),
                "location_label": ref.get("location_label"),
                "location_id": ref.get("location_id"),
                "participant_person_ids": sorted(set(ref.get("participant_person_ids") or [])),
                "participant_labels": sorted(set(ref.get("participant_labels") or [])),
                "event_kind": ref.get("event_kind") if is_document else None,
                "event_class": ref.get("event_class"),
                "support_count": 0,
                "supporting_sources": [],
                "merge_method": "canonical_or_conservative_key",
                "merge_key": key,
            }
            clusters.append(match)
        match["support_count"] += 1
        match["supporting_sources"].append(
            {
                "source_id": ref.get("source_id"),
                "source_event_ref_id": ref.get("source_event_ref_id"),
                "page_refs": ref.get("page_refs") or [],
                "supporting_quote": ref.get("supporting_quote"),
                "summary": ref.get("summary"),
                "confidence": ref.get("confidence"),
            }
        )
        for person_id in ref.get("participant_person_ids") or []:
            if person_id not in match["participant_person_ids"]:
                match["participant_person_ids"].append(person_id)
                match["participant_person_ids"].sort()
    return clusters


def conservative_event_key(ref: dict[str, Any], *, is_document: bool) -> str:
    event_time = ref.get("event_time") or {}
    parts = [
        ref.get("canonical_id") or "",
        event_time.get("normalized_date") or event_time.get("start") or "",
        normalize_key(ref.get("label")),
        normalize_key(ref.get("location_label")),
        ref.get("event_kind") if is_document else "",
    ]
    return "|".join(str(part or "") for part in parts)


def find_cluster(
    ref: dict[str, Any],
    key: str,
    clusters: list[dict[str, Any]],
    *,
    is_document: bool,
    approved_pairs: set[frozenset[str]],
) -> dict[str, Any] | None:
    canonical_id = ref.get("canonical_id")
    ref_id = ref.get("source_event_ref_id")
    date = ((ref.get("event_time") or {}).get("normalized_date") or "")[:10]
    location = normalize_key(ref.get("location_label"))
    label = normalize_key(ref.get("label"))
    for cluster in clusters:
        if ref_id and any(
            frozenset((ref_id, support.get("source_event_ref_id"))) in approved_pairs
            for support in cluster.get("supporting_sources", [])
            if support.get("source_event_ref_id")
        ):
            cluster["merge_method"] = "semantic_or_llm_approved_pair"
            return cluster
        if canonical_id and cluster.get("canonical_id") == canonical_id:
            return cluster
        if cluster.get("merge_key") == key:
            return cluster
        cluster_date = (((cluster.get("event_time") or {}).get("normalized_date") or "")[:10])
        cluster_location = normalize_key(cluster.get("location_label"))
        cluster_label = normalize_key(cluster.get("label"))
        same_context = bool(date and cluster_date == date and location == cluster_location)
        same_kind = (not is_document) or ((cluster.get("event_kind") or "") == (ref.get("event_kind") or ""))
        if same_context and same_kind and labels_are_near_duplicates(label, cluster_label):
            cluster["merge_method"] = "embedding_cosine_or_character_similarity"
            return cluster
    return None


def merge_aligned_event_clusters(
    clusters: list[dict[str, Any]],
    *,
    vectors_by_id: dict[str, list[float]],
    refs_by_id: dict[str, dict[str, Any]],
    is_document: bool,
) -> dict[str, Any]:
    if len(clusters) < 2:
        return {"input_clusters": len(clusters), "output_clusters": len(clusters), "merged_clusters": 0, "matches": []}
    profiles = {
        cluster["id"]: cluster_similarity_profile(cluster, vectors_by_id=vectors_by_id, refs_by_id=refs_by_id)
        for cluster in clusters
    }
    parent = {cluster["id"]: cluster["id"] for cluster in clusters}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    matches: list[dict[str, Any]] = []
    for left_index, left in enumerate(clusters):
        for right in clusters[left_index + 1 :]:
            decision = score_cluster_merge_candidate(
                left,
                right,
                left_profile=profiles[left["id"]],
                right_profile=profiles[right["id"]],
                is_document=is_document,
            )
            if not decision["merge"]:
                continue
            union(left["id"], right["id"])
            matches.append(decision)

    groups: dict[str, list[dict[str, Any]]] = {}
    for cluster in clusters:
        groups.setdefault(find(cluster["id"]), []).append(cluster)
    merged = [merge_cluster_group(group) for group in groups.values()]
    clusters[:] = sorted(merged, key=lambda item: (event_date_sort_key(item), normalize_key(item.get("label")), item.get("id") or ""))
    return {
        "input_clusters": len(parent),
        "output_clusters": len(clusters),
        "merged_clusters": len(parent) - len(clusters),
        "matches": matches,
    }


def cluster_similarity_profile(
    cluster: dict[str, Any],
    *,
    vectors_by_id: dict[str, list[float]],
    refs_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    support_ref_ids = [
        support.get("source_event_ref_id")
        for support in cluster.get("supporting_sources") or []
        if support.get("source_event_ref_id")
    ]
    vectors = [
        vectors_by_id[refs_by_id[ref_id]["embedding_id"]]
        for ref_id in support_ref_ids
        if ref_id in refs_by_id
        and refs_by_id[ref_id].get("embedding_id") in vectors_by_id
    ]
    return {
        "text": cluster_identity_text(cluster),
        "vector": average_vectors(vectors),
        "location_tokens": distinctive_location_tokens(cluster.get("location_label")),
        "label_tokens": significant_tokens(normalize_key(cluster.get("label"))),
    }


def cluster_identity_text(cluster: dict[str, Any]) -> str:
    supports = []
    for support in (cluster.get("supporting_sources") or [])[:24]:
        supports.append(
            {
                "summary": shorten_quote(support.get("summary"), max_length=260),
                "quote": shorten_quote(support.get("supporting_quote"), max_length=180),
            }
        )
    payload = {
        "label": cluster.get("label"),
        "date": (cluster.get("event_time") or {}).get("normalized_date") or (cluster.get("event_time") or {}).get("start"),
        "location": cluster.get("location_label"),
        "participants": cluster.get("participant_labels") or cluster.get("participant_person_ids") or [],
        "event_kind": cluster.get("event_kind"),
        "event_class": cluster.get("event_class"),
        "supports": supports,
    }
    return json_dumps(payload)


def average_vectors(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    dimensions = len(vectors[0])
    compatible = [vector for vector in vectors if len(vector) == dimensions]
    if not compatible:
        return None
    return [sum(vector[index] for vector in compatible) / len(compatible) for index in range(dimensions)]


def score_cluster_merge_candidate(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_profile: dict[str, Any],
    right_profile: dict[str, Any],
    is_document: bool,
) -> dict[str, Any]:
    left_label = normalize_key(left.get("label"))
    right_label = normalize_key(right.get("label"))
    canonical_match = bool(left.get("canonical_id") and left.get("canonical_id") == right.get("canonical_id"))
    date_compatible = dates_compatible(left, right)
    exact_date_match = cluster_date_value(left) != "" and cluster_date_value(left) == cluster_date_value(right)
    kind_compatible = (not is_document) or ((left.get("event_kind") or "") == (right.get("event_kind") or ""))
    if not canonical_match and (not exact_date_match or not kind_compatible):
        return {
            "merge": False,
            "left_cluster_id": left.get("id"),
            "right_cluster_id": right.get("id"),
            "left_label": left.get("label"),
            "right_label": right.get("label"),
            "reason": "",
            "scores": {},
            "constraints": {
                "date_compatible": date_compatible,
                "exact_date_match": exact_date_match,
                "place_compatible": None,
                "kind_compatible": kind_compatible,
                "shared_historical_terms": [],
                "distinctive_place_overlap": False,
                "high_impact_anchor": False,
            },
        }
    place_compatible = cluster_places_compatible(left, right, left_profile, right_profile)
    if not canonical_match and not place_compatible:
        return {
            "merge": False,
            "left_cluster_id": left.get("id"),
            "right_cluster_id": right.get("id"),
            "left_label": left.get("label"),
            "right_label": right.get("label"),
            "reason": "",
            "scores": {},
            "constraints": {
                "date_compatible": date_compatible,
                "exact_date_match": exact_date_match,
                "place_compatible": place_compatible,
                "kind_compatible": kind_compatible,
                "shared_historical_terms": [],
                "distinctive_place_overlap": False,
                "high_impact_anchor": False,
            },
        }
    label_score = max(
        SequenceMatcher(None, left_label, right_label).ratio(),
        cosine_similarity(left_label, right_label),
        token_overlap(left_label, right_label),
    )
    full_text_score = cosine_similarity(left_profile["text"], right_profile["text"])
    embedding_score = vector_cosine(left_profile.get("vector"), right_profile.get("vector"))
    semantic_score = max(full_text_score, embedding_score or 0.0)
    shared_terms = left_profile["label_tokens"].intersection(right_profile["label_tokens"]).intersection(HISTORICAL_TERMS)
    distinctive_place_overlap = bool(left_profile["location_tokens"].intersection(right_profile["location_tokens"]))
    high_impact = has_high_impact_anchor(left) and has_high_impact_anchor(right)
    merge = False
    reason = ""
    if canonical_match:
        merge = True
        reason = "same canonical id"
    elif exact_date_match and place_compatible and kind_compatible and labels_are_near_duplicates(left_label, right_label):
        merge = True
        reason = "same date/place/kind and near-duplicate labels"
    elif (
        exact_date_match
        and place_compatible
        and kind_compatible
        and shared_terms
        and (semantic_score >= (0.68 if not is_document else 0.78) or label_score >= 0.74)
    ):
        merge = True
        reason = "same date/place/kind with aligned full-object semantics"
    elif (
        exact_date_match
        and kind_compatible
        and high_impact
        and distinctive_place_overlap
        and shared_terms
        and semantic_score >= (0.64 if not is_document else 0.76)
    ):
        merge = True
        reason = "high-impact same-date event with distinctive place and aligned semantics"
    return {
        "merge": merge,
        "left_cluster_id": left.get("id"),
        "right_cluster_id": right.get("id"),
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "reason": reason,
        "scores": {
            "label_similarity": round(label_score, 4),
            "full_object_similarity": round(full_text_score, 4),
            "aggregate_embedding_cosine": round(embedding_score, 4) if embedding_score is not None else None,
        },
        "constraints": {
            "date_compatible": date_compatible,
            "exact_date_match": exact_date_match,
            "place_compatible": place_compatible,
            "kind_compatible": kind_compatible,
            "shared_historical_terms": sorted(shared_terms),
            "distinctive_place_overlap": distinctive_place_overlap,
            "high_impact_anchor": high_impact,
        },
    }


def merge_cluster_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    if len(group) == 1:
        return group[0]
    group = sorted(group, key=lambda item: (-int(item.get("support_count") or 0), len(str(item.get("label") or ""))))
    merged = copy.deepcopy(group[0])
    support_by_ref = {
        support.get("source_event_ref_id"): support
        for support in merged.get("supporting_sources") or []
        if support.get("source_event_ref_id")
    }
    for cluster in group[1:]:
        for support in cluster.get("supporting_sources") or []:
            ref_id = support.get("source_event_ref_id")
            if ref_id and ref_id in support_by_ref:
                continue
            merged.setdefault("supporting_sources", []).append(support)
            if ref_id:
                support_by_ref[ref_id] = support
        for person_id in cluster.get("participant_person_ids") or []:
            if person_id not in merged.setdefault("participant_person_ids", []):
                merged["participant_person_ids"].append(person_id)
        for label in cluster.get("participant_labels") or []:
            if label not in merged.setdefault("participant_labels", []):
                merged["participant_labels"].append(label)
    merged["participant_person_ids"] = sorted(merged.get("participant_person_ids") or [])
    merged["participant_labels"] = sorted(merged.get("participant_labels") or [])
    merged["support_count"] = len(merged.get("supporting_sources") or [])
    merged["merge_method"] = "cluster_second_pass_full_object_similarity"
    merged["merged_cluster_ids"] = sorted(cluster.get("id") for cluster in group if cluster.get("id"))
    merged["id"] = rebuild_cluster_id_from_merged(merged)
    return merged


def rebuild_cluster_id_from_merged(cluster: dict[str, Any]) -> str:
    prefix = "brief_document_event" if cluster.get("event_kind") else "brief_event"
    support_ids = sorted(
        support.get("source_event_ref_id")
        for support in cluster.get("supporting_sources") or []
        if support.get("source_event_ref_id")
    )
    seed = "|".join([cluster.get("label") or "", *support_ids])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    base = " ".join(
        str(part)
        for part in [
            (cluster.get("event_time") or {}).get("normalized_date"),
            cluster.get("label"),
            cluster.get("location_label"),
        ]
        if part
    )
    return f"{slugify(normalize_key(base), prefix)}_{digest}"


def event_date_sort_key(cluster: dict[str, Any]) -> str:
    return (((cluster.get("event_time") or {}).get("normalized_date") or (cluster.get("event_time") or {}).get("start") or ""))


def cluster_date_value(cluster: dict[str, Any]) -> str:
    return str(((cluster.get("event_time") or {}).get("normalized_date") or (cluster.get("event_time") or {}).get("start") or ""))[:10]


def cluster_places_compatible(
    left: dict[str, Any],
    right: dict[str, Any],
    left_profile: dict[str, Any],
    right_profile: dict[str, Any],
) -> bool:
    if places_compatible(left, right):
        return True
    left_tokens = left_profile.get("location_tokens") or set()
    right_tokens = right_profile.get("location_tokens") or set()
    if not left_tokens or not right_tokens:
        return True
    return bool(left_tokens.intersection(right_tokens))


def labels_are_near_duplicates(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_terms = set(left.split())
    right_terms = set(right.split())
    shared_historical_terms = left_terms.intersection(right_terms).intersection(HISTORICAL_TERMS)
    if "haymarket" in shared_historical_terms and {"bomb", "bombing", "explosion"}.intersection(left_terms | right_terms):
        return True
    return max(SequenceMatcher(None, left, right).ratio(), cosine_similarity(left, right)) >= 0.72


def build_cluster_id(prefix: str, ref: dict[str, Any], index: int) -> str:
    event_time = ref.get("event_time") or {}
    parts = [
        ref.get("canonical_id"),
        event_time.get("normalized_date") or event_time.get("start"),
        ref.get("label"),
        ref.get("location_label"),
    ]
    key = normalize_key(" ".join(str(part) for part in parts if part))
    if not key:
        key = f"cluster_{index}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(key, prefix)}_{digest}"


def assign_cluster_ids(
    harmonized_by_source: dict[str, dict[str, Any]],
    clusters: list[dict[str, Any]],
    event_field: str,
    id_field: str,
) -> None:
    cluster_by_ref = {
        support.get("source_event_ref_id"): cluster.get("id")
        for cluster in clusters
        for support in cluster.get("supporting_sources", [])
        if support.get("source_event_ref_id")
    }
    for briefing in harmonized_by_source.values():
        for ref in briefing.get(event_field) or []:
            ref_id = ref.get("source_event_ref_id")
            if ref_id in cluster_by_ref:
                ref[id_field] = cluster_by_ref[ref_id]


def build_coverage(
    *,
    pages: list[dict[str, Any]],
    briefings_by_source: dict[str, dict[str, Any]],
    harmonized_by_source: dict[str, dict[str, Any]],
    event_clusters: list[dict[str, Any]],
    document_event_clusters: list[dict[str, Any]],
    routine_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_referenced = sum(len((briefing or {}).get("referenced_events") or []) for briefing in briefings_by_source.values())
    raw_document = sum(len((briefing or {}).get("document_events") or []) for briefing in briefings_by_source.values())
    harmonized_referenced = sum(len((briefing or {}).get("referenced_events") or []) for briefing in harmonized_by_source.values())
    harmonized_document = sum(len((briefing or {}).get("document_events") or []) for briefing in harmonized_by_source.values())
    cluster_supports = sum(cluster.get("support_count", 0) for cluster in event_clusters)
    document_cluster_supports = sum(cluster.get("support_count", 0) for cluster in document_event_clusters)
    covered_ref_ids = {
        support.get("source_event_ref_id")
        for cluster in [*event_clusters, *document_event_clusters]
        for support in cluster.get("supporting_sources", [])
        if support.get("source_event_ref_id")
    }
    covered_ref_ids.update(ref.get("source_event_ref_id") for ref in routine_refs if ref.get("source_event_ref_id"))
    covered = len(covered_ref_ids)
    expected = raw_referenced + raw_document
    return {
        "pages": len(pages),
        "raw_brief_sources": len(briefings_by_source),
        "harmonized_sources": len(harmonized_by_source),
        "missing_brief_sources": sorted({page.get("id") for page in pages if page.get("id") not in briefings_by_source}),
        "raw_event_refs": raw_referenced,
        "raw_document_event_refs": raw_document,
        "harmonized_event_refs": harmonized_referenced,
        "harmonized_document_event_refs": harmonized_document,
        "routine_procedural_refs": len(routine_refs),
        "event_clusters": len(event_clusters),
        "document_event_clusters": len(document_event_clusters),
        "covered_refs": covered,
        "expected_refs": expected,
        "coverage_ok": covered == expected,
    }


def dedupe_refs(refs: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        key = "|".join(str(ref.get(part) or "") for part in keys)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def normalize_page_refs(values: list[Any]) -> list[str]:
    refs: list[str] = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        match = re.search(r"(?:pp?\.?|pages?)\s+([A-Z]?\s?\d+(?:\s+1/2)?)", text, flags=re.I)
        ref = match.group(1) if match else text
        ref = clean_text(ref)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def shorten_quote(value: Any, max_length: int = 220) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "..."


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def embedding_cache_key(item: dict[str, Any], model: str, dimensions: int) -> str:
    payload = "|".join(
        [
            HARMONIZATION_VERSION,
            model,
            str(dimensions),
            item.get("kind") or "",
            item.get("text_sha256") or "",
        ]
    )
    return sha256_text(payload)


def embedding_cache_path(cache_key: str) -> str:
    return f"cache/haymarket/brief_harmonization_embeddings/{cache_key}.json"


def zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "cached_input_tokens": 0}


def add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in total:
        total[key] += int((usage or {}).get(key) or 0)


def chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def event_date_text(value: Any) -> str:
    if not isinstance(value, dict):
        return clean_text(value)
    return clean_text(
        value.get("normalized_date")
        or value.get("start")
        or value.get("display")
        or value.get("original_text")
    )


def candidate_group(ref_type: str) -> str:
    if ref_type in {"historical_event", "document_event", "routine_procedural"}:
        return "event"
    if ref_type in {"place", "canonical_place"}:
        return "place"
    if ref_type in {"person", "canonical_person"}:
        return "person"
    return ref_type


def vector_cosine(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return None
    return dot / (left_norm * right_norm)


def token_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_key(left).split())
    right_tokens = set(normalize_key(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


def dates_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_date = (((left.get("event_time") or {}).get("normalized_date") or "")[:10])
    right_date = (((right.get("event_time") or {}).get("normalized_date") or "")[:10])
    return not left_date or not right_date or left_date == right_date


def places_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_id = left.get("location_id")
    right_id = right.get("location_id")
    if left_id and right_id:
        return left_id == right_id
    left_key = normalize_key(left.get("location_label"))
    right_key = normalize_key(right.get("location_label"))
    if not left_key or not right_key:
        return True
    return left_key == right_key or labels_are_near_duplicates(left_key, right_key)


def participant_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_people = set(left.get("participant_person_ids") or [])
    right_people = set(right.get("participant_person_ids") or [])
    if left_people and right_people:
        return len(left_people.intersection(right_people)) / len(left_people.union(right_people))
    left_labels = {normalize_key(label) for label in left.get("participant_labels") or [] if normalize_key(label)}
    right_labels = {normalize_key(label) for label in right.get("participant_labels") or [] if normalize_key(label)}
    if left_labels and right_labels:
        return len(left_labels.intersection(right_labels)) / len(left_labels.union(right_labels))
    return 0.5 if not left_people and not right_people and not left_labels and not right_labels else 0.0


def has_high_impact_anchor(ref: dict[str, Any]) -> bool:
    text = normalize_key(" ".join(str(ref.get(key) or "") for key in ("label", "description", "source_title", "location_label")))
    anchors = ("haymarket", "mccormick", "revenge", "arbeiter zeitung", "greif", "zepf", "bomb", "dynamite")
    return any(anchor in text for anchor in anchors)


def is_high_risk_match(match: dict[str, Any]) -> bool:
    constraints = match.get("constraints") or {}
    if constraints.get("high_impact_anchor"):
        return True
    if not constraints.get("date_compatible") or not constraints.get("place_compatible"):
        return True
    return float(match.get("overall_score") or 0.0) < 0.84


def llm_reviewable_match(match: dict[str, Any]) -> bool:
    if match.get("decision") != "needs_llm_review":
        return False
    group = match.get("group")
    if group not in {"event", "person", "place"}:
        return False
    if match.get("right_is_canonical_anchor") and group != "event":
        return False
    return True


def llm_review_priority(match: dict[str, Any]) -> tuple[int, int, float, str]:
    group = match.get("group")
    constraints = match.get("constraints") or {}
    group_rank = 0 if group == "event" else 1
    anchor_rank = 0 if constraints.get("high_impact_anchor") else 1
    return (group_rank, anchor_rank, -float(match.get("overall_score") or 0.0), str(match.get("pair_id") or ""))


def pair_id_for(left_ref_id: str, right_ref_id: str) -> str:
    ordered = sorted([left_ref_id, right_ref_id])
    return f"pair_{sha256_text('|'.join(ordered))[:16]}"


def compact_ref_for_review(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref_id": ref.get("ref_id"),
        "type": ref.get("ref_type"),
        "label": ref.get("label"),
        "canonical_id": ref.get("canonical_id"),
        "date": (ref.get("event_time") or {}).get("normalized_date"),
        "location": ref.get("location_label"),
        "participants": ref.get("participant_labels") or ref.get("participant_person_ids") or [],
        "summary": shorten_quote(ref.get("description"), max_length=260),
        "quote": shorten_quote(ref.get("supporting_quote"), max_length=180),
        "source_title": ref.get("source_title"),
    }


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cosine_similarity(left: str, right: str) -> float:
    left_vec = char_ngram_vector(left)
    right_vec = char_ngram_vector(right)
    if not left_vec or not right_vec:
        return 0.0
    overlap = set(left_vec).intersection(right_vec)
    dot = sum(left_vec[key] * right_vec[key] for key in overlap)
    left_norm = math.sqrt(sum(value * value for value in left_vec.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vec.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def char_ngram_vector(value: str, n: int = 3) -> Counter[str]:
    text = f"  {normalize_key(value)}  "
    if len(text) < n:
        return Counter({text: 1}) if text.strip() else Counter()
    return Counter(text[index : index + n] for index in range(0, len(text) - n + 1))
