from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from typing import Any

from sources.hadc_source import XML_NS, serialize_xml
from utils.ids import slugify


def harmonize_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    people, person_map = harmonize_entities(collect(bundles, "people"), "person", person_key, "display_name")
    locations, location_map = harmonize_entities(collect(bundles, "locations"), "location", location_key, "name")
    events, event_map = harmonize_entities(collect(bundles, "event_suggestions"), "event", event_key, "title")

    id_maps = {
        "people": person_map,
        "locations": location_map,
        "events": event_map,
    }
    claims = [rewrite_claim(claim, id_maps) for claim in collect(bundles, "claims")]
    quotes = [rewrite_quote(quote, id_maps) for quote in collect(bundles, "quotes")]
    rewritten_bundles = [rewrite_bundle(bundle, id_maps) for bundle in bundles]

    return {
        "people": enhance_people(people),
        "locations": enhance_locations(locations),
        "events": enhance_events(events),
        "claims": claims,
        "quotes": quotes,
        "bundles": rewritten_bundles,
        "id_maps": id_maps,
        "merge_counts": {
            "people": len(collect(bundles, "people")) - len(people),
            "locations": len(collect(bundles, "locations")) - len(locations),
            "events": len(collect(bundles, "event_suggestions")) - len(events),
        },
    }


def collect(bundles: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for bundle in bundles:
        items.extend(bundle.get(key, []))
    return items


def harmonize_entities(items: list[dict[str, Any]], prefix: str, key_func, label_field: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[str, dict[str, Any]] = {}
    id_map: dict[str, str] = {}
    for item in items:
        key = key_func(item)
        if not key:
            continue
        canonical_id = slugify(key, prefix)
        incoming = copy.deepcopy(item)
        old_id = incoming.get("id")
        incoming["id"] = canonical_id
        if old_id:
            id_map[old_id] = canonical_id
        id_map[canonical_id] = canonical_id
        if canonical_id not in grouped:
            grouped[canonical_id] = incoming
        else:
            grouped[canonical_id] = merge_item(grouped[canonical_id], incoming)

        if label_field in grouped[canonical_id] and not grouped[canonical_id].get(label_field):
            grouped[canonical_id][label_field] = item.get(label_field)

    return sorted(grouped.values(), key=lambda item: item["id"]), id_map


def merge_item(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(existing)
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


def person_key(person: dict[str, Any]) -> str:
    names = [person.get("display_name"), *person.get("alternate_names", [])]
    return normalize_label(next((name for name in names if name), person.get("id", "")))


def location_key(location: dict[str, Any]) -> str:
    labels = [location.get("name"), location.get("address_1886"), location.get("modern_address"), location.get("id")]
    return normalize_label(next((label for label in labels if label), ""))


def event_key(event: dict[str, Any]) -> str:
    time = event.get("time") or {}
    parts = [event.get("title"), time.get("start"), event.get("location_id")]
    return normalize_label(" ".join(str(part) for part in parts if part))


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def rewrite_bundle(bundle: dict[str, Any], id_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = copy.deepcopy(bundle)
    rewritten["people"] = [rewrite_person(person, id_maps) for person in bundle.get("people", [])]
    rewritten["locations"] = [rewrite_location(location, id_maps) for location in bundle.get("locations", [])]
    rewritten["claims"] = [rewrite_claim(claim, id_maps) for claim in bundle.get("claims", [])]
    rewritten["event_suggestions"] = [rewrite_event(event, id_maps) for event in bundle.get("event_suggestions", [])]
    rewritten["quotes"] = [rewrite_quote(quote, id_maps) for quote in bundle.get("quotes", [])]
    if bundle.get("tei_xml"):
        rewritten["tei_xml"] = rewrite_tei_refs(bundle["tei_xml"], id_maps)
    return rewritten


def rewrite_person(person: dict[str, Any], id_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = copy.deepcopy(person)
    rewritten["id"] = rewrite_id(rewritten.get("id"), id_maps["people"])
    return rewritten


def rewrite_location(location: dict[str, Any], id_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = copy.deepcopy(location)
    rewritten["id"] = rewrite_id(rewritten.get("id"), id_maps["locations"])
    return rewritten


def rewrite_claim(claim: dict[str, Any], id_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = copy.deepcopy(claim)
    rewritten["reported_by_person_id"] = rewrite_id(rewritten.get("reported_by_person_id"), id_maps["people"])
    rewritten["subject_person_ids"] = rewrite_ids(rewritten.get("subject_person_ids", []), id_maps["people"])
    rewritten["location_id"] = rewrite_id(rewritten.get("location_id"), id_maps["locations"])
    return rewritten


def rewrite_event(event: dict[str, Any], id_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = copy.deepcopy(event)
    rewritten["id"] = rewrite_id(rewritten.get("id"), id_maps["events"])
    rewritten["location_id"] = rewrite_id(rewritten.get("location_id"), id_maps["locations"])
    rewritten["participant_person_ids"] = rewrite_ids(rewritten.get("participant_person_ids", []), id_maps["people"])
    return rewritten


def rewrite_quote(quote: dict[str, Any], id_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    rewritten = copy.deepcopy(quote)
    rewritten["speaker_person_id"] = rewrite_id(rewritten.get("speaker_person_id"), id_maps["people"])
    return rewritten


def rewrite_id(value: Any, id_map: dict[str, str]) -> Any:
    if value is None:
        return None
    return id_map.get(str(value), value)


def rewrite_ids(values: list[Any], id_map: dict[str, str]) -> list[str]:
    return sorted({str(rewrite_id(value, id_map)) for value in values if value is not None})


def rewrite_tei_refs(tei_xml: str, id_maps: dict[str, dict[str, str]]) -> str:
    all_maps: dict[str, str] = {}
    for mapping in id_maps.values():
        all_maps.update(mapping)
    root = ET.fromstring(tei_xml)
    ref_attrs = {"who", "corresp", "resp", "source", "ana", "ref"}
    for element in root.iter():
        for attr in ref_attrs:
            value = element.attrib.get(attr)
            if value:
                element.attrib[attr] = rewrite_ref_value(value, all_maps)
        xml_id = element.attrib.get(f"{{{XML_NS}}}id")
        if xml_id and xml_id in all_maps and not element.attrib.get("type") == "provisional":
            element.attrib[f"{{{XML_NS}}}id"] = all_maps[xml_id]
    return serialize_xml(root)


def rewrite_ref_value(value: str, id_map: dict[str, str]) -> str:
    refs = value.split()
    rewritten = []
    for ref in refs:
        prefix = "#" if ref.startswith("#") else ""
        clean = ref.lstrip("#")
        rewritten.append(f"{prefix}{id_map.get(clean, clean)}")
    return " ".join(rewritten)


def enhance_people(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for person in people:
        person["source_ids"] = sorted(set(person.get("source_ids", [])))
        person["roles"] = sorted(set(person.get("roles", [])))
    return people


def enhance_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for location in locations:
        location["source_ids"] = sorted(set(location.get("source_ids", [])))
        coordinates = location.setdefault("coordinates", {})
        coordinates.setdefault("lat", None)
        coordinates.setdefault("lng", None)
        coordinates.setdefault("confidence", 0)
        coordinates.setdefault("method", "tei_llm_unresolved")
    return locations


def enhance_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for event in events:
        event["participant_person_ids"] = sorted(set(event.get("participant_person_ids", [])))
        event["claim_ids"] = sorted(set(event.get("claim_ids", [])))
    return events
