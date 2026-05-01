from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from utils.ids import slugify
from utils.s3_storage import JsonStorage


HARMONIZATION_VERSION = "brief_harmonization_v1"
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
) -> dict[str, Any]:
    result = harmonize_briefings(
        pages=pages,
        briefings_by_source=briefings_by_source,
        run_id=run_id,
        people=people or [],
        locations=locations or [],
        events=events or [],
    )
    storage.write_json("enriched/haymarket/brief_harmonization/latest.json", result["artifact"])
    storage.write_json(f"raw/haymarket/brief_harmonization/{run_id}/audit.json", result["artifact"])
    return result


def harmonize_briefings(
    *,
    pages: list[dict[str, Any]],
    briefings_by_source: dict[str, dict[str, Any]],
    run_id: str,
    people: list[dict[str, Any]] | None = None,
    locations: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
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

    event_clusters = cluster_event_refs(event_refs, "brief_event", is_document=False)
    document_event_clusters = cluster_event_refs(document_event_refs, "brief_document_event", is_document=True)
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
            "embedding_cosine": "local_char_ngram_cosine",
            "llm_confirmation": "deferred_to_review_layer",
        },
    }
    return {
        "briefings_by_source": harmonized_by_source,
        "artifact": artifact,
        "coverage": coverage,
        "event_clusters": event_clusters,
        "document_event_clusters": document_event_clusters,
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


def cluster_event_refs(refs: list[dict[str, Any]], prefix: str, *, is_document: bool) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for ref in refs:
        key = conservative_event_key(ref, is_document=is_document)
        match = find_cluster(ref, key, clusters, is_document=is_document)
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
) -> dict[str, Any] | None:
    canonical_id = ref.get("canonical_id")
    date = ((ref.get("event_time") or {}).get("normalized_date") or "")[:10]
    location = normalize_key(ref.get("location_label"))
    label = normalize_key(ref.get("label"))
    for cluster in clusters:
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
