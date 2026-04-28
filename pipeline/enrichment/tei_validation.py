from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Any

from sources.hadc_source import TEI_NS, XML_NS, parse_tei_xml, tei_tag, tei_to_transcript_json


TEXT_DRIFT_MIN_RATIO = 0.92
TEXT_TOKEN_RECALL_MIN = 0.95


class TEIValidationError(Exception):
    def __init__(self, message: str, validation: dict[str, Any]) -> None:
        super().__init__(message)
        self.validation = validation


def validate_generated_tei(page: dict[str, Any], tei_xml: str) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "status": "valid",
        "errors": [],
        "warnings": [],
    }
    try:
        root = parse_tei_xml(tei_xml)
    except ET.ParseError as exc:
        validation["status"] = "invalid"
        validation["errors"].append(f"XML parse error: {exc}")
        return validation

    body = root.find(f".//{tei_tag('body')}")
    if body is None:
        validation["status"] = "invalid"
        validation["errors"].append("Missing TEI body")

    candidate_text = page.get("text") or ""
    transcript = tei_to_transcript_json(
        source_id=page["id"],
        url=page["url"],
        title=page["title"],
        source_type=page["source_type"],
        fetched_at=page["fetched_at"],
        tei_xml=tei_xml,
        transcript_metadata=page.get("transcript_metadata", {}),
    )
    tei_text = transcript["text"]
    candidate_norm = normalize_for_text_validation(candidate_text)
    tei_norm = normalize_for_text_validation(tei_text)
    ratio = SequenceMatcher(None, candidate_norm, tei_norm).ratio() if candidate_norm or tei_norm else 1.0
    tokens_candidate = set(re.findall(r"\w+", candidate_norm))
    tokens_tei = set(re.findall(r"\w+", tei_norm))
    token_recall = (
        len(tokens_candidate & tokens_tei) / len(tokens_candidate)
        if tokens_candidate
        else 1.0
    )
    validation["text"] = {
        "candidate_chars": len(candidate_text),
        "tei_chars": len(tei_text),
        "candidate_normalized_chars": len(candidate_norm),
        "tei_normalized_chars": len(tei_norm),
        "similarity_ratio": round(ratio, 4),
        "token_recall": round(token_recall, 4),
    }
    if ratio < TEXT_DRIFT_MIN_RATIO and token_recall < TEXT_TOKEN_RECALL_MIN:
        validation["status"] = "invalid"
        validation["errors"].append(
            f"TEI text drift ratio {ratio:.4f} below {TEXT_DRIFT_MIN_RATIO:.2f} "
            f"and token recall {token_recall:.4f} below {TEXT_TOKEN_RECALL_MIN:.2f}"
        )

    expected_page_refs = {str(cue.get("page_ref")) for cue in page.get("page_cues", []) if cue.get("page_ref")}
    actual_page_refs = {str(ref.get("page_ref")) for ref in transcript.get("page_refs", []) if ref.get("page_ref")}
    missing_page_refs = sorted(expected_page_refs - actual_page_refs)
    validation["page_markers"] = {
        "expected": len(expected_page_refs),
        "actual": len(actual_page_refs),
        "missing": missing_page_refs,
    }
    if missing_page_refs:
        validation["warnings"].append(f"Missing page refs in TEI: {', '.join(missing_page_refs[:10])}")

    validation["speaker_attribution"] = speaker_attribution_stats(root)
    validation["annotation_targets"] = annotation_target_stats(root)
    if validation["annotation_targets"]["missing"]:
        validation["status"] = "invalid"
        validation["errors"].append("One or more TEI annotation targets do not resolve")

    return validation


def validate_generated_tei_or_raise(page: dict[str, Any], tei_xml: str) -> dict[str, Any]:
    validation = validate_generated_tei(page, tei_xml)
    if validation["status"] != "valid":
        raise TEIValidationError("Generated TEI failed validation", validation)
    return validation


def normalize_for_text_validation(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if re.match(r"^\[Image,\s*.+?\]$", line.strip()):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", "\n".join(lines)).strip().lower()


def speaker_attribution_stats(root: ET.Element) -> dict[str, int]:
    speech = root.findall(f".//{tei_tag('sp')}")
    answer_turns = [turn for turn in speech if turn.attrib.get("type") == "answer"]
    answer_with_who = [turn for turn in answer_turns if turn.attrib.get("who")]
    turns_with_who = [turn for turn in speech if turn.attrib.get("who")]
    return {
        "speech_turns": len(speech),
        "turns_with_who": len(turns_with_who),
        "answer_turns": len(answer_turns),
        "answer_turns_with_who": len(answer_with_who),
    }


def annotation_target_stats(root: ET.Element) -> dict[str, Any]:
    ids = {
        element.attrib.get(f"{{{XML_NS}}}id") or element.attrib.get("id")
        for element in root.iter()
        if element.attrib.get(f"{{{XML_NS}}}id") or element.attrib.get("id")
    }
    annotations = root.findall(f".//{tei_tag('annotation')}")
    missing = []
    for annotation in annotations:
        target = annotation.attrib.get("target", "")
        if not target or re.match(r"#char-\d+-\d+$", target):
            continue
        target_id = target.lstrip("#")
        if target_id not in ids:
            missing.append(target_id)
    return {"annotations": len(annotations), "missing": missing}
