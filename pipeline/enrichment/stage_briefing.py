from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from enrichment.progress import StageProgress
from utils.ids import slugify
from utils.openai_schema import (
    LLMCallError,
    call_openai_structured,
    estimate_cost_usd,
    load_schema_with_refs,
)
from utils.s3_storage import JsonStorage


BRIEFING_PROMPT_TEMPLATE = "haymarket_page_briefing_v4"
# Bumped from 2k to 8k so reasoning models (gpt-5 family) have room for both
# reasoning tokens and the structured answer. Briefing identifying every
# speaker on a long page benefits from reasoning, but the budget needs
# headroom or the JSON answer gets truncated like Stage C did.
BRIEFING_MAX_OUTPUT_TOKENS = 8_000
# Briefing asks the model to find every named speaker, canonicalize names,
# infer roles — a reasoning-friendly task. "low" effort preserves output
# budget while still giving the model a few hundred tokens to think.
BRIEFING_REASONING_EFFORT = "low"


def run_briefing(
    page: dict[str, Any],
    storage: JsonStorage,
    run_id: str,
    model: str = "gpt-5-mini",
    progress: StageProgress | None = None,
) -> dict[str, Any]:
    progress = progress or StageProgress(page["id"], "briefing", enabled=False)
    progress.info(f"starting ({model})")
    start = time.monotonic()

    schema = load_briefing_schema()
    messages = build_briefing_messages(page)

    try:
        parsed, raw_output, usage = call_openai_structured(
            model=model,
            input_messages=messages,
            schema=schema,
            schema_name="haymarket_page_briefing",
            max_output_tokens=BRIEFING_MAX_OUTPUT_TOKENS,
            reasoning_effort=BRIEFING_REASONING_EFFORT,
        )
        # Force speaker_directory IDs into canonical person_<slug> form so
        # downstream stages (TEI <sp who>, person.id in tagging) share one
        # vocabulary. The model's exact id choice doesn't matter — we
        # rewrite from display_name.
        normalize_speaker_directory(parsed)
        status = "success"
        error = None
    except Exception as exc:
        parsed = None
        raw_output = exc.raw_output if isinstance(exc, LLMCallError) else None
        usage = exc.usage if isinstance(exc, LLMCallError) else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        status = "error"
        error = str(exc)

    cost_usd = estimate_cost_usd(model, usage)
    duration = time.monotonic() - start

    audit = {
        "run_id": run_id,
        "page_id": page["id"],
        "stage": "briefing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": BRIEFING_PROMPT_TEMPLATE,
        "input_messages": messages,
        "raw_output": raw_output,
        "parsed_output": parsed,
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
        "status": status,
        "error": error,
        "duration_s": round(duration, 3),
    }

    audit_path = f"raw/haymarket/llm/{run_id}/{slugify(model)}/{page['id']}/briefing.json"
    storage.write_json(audit_path, audit)

    if status == "success":
        progress.done(
            f"{briefing_log_label(page, parsed)}, ${cost_usd:.4f}",
            duration_s=duration,
        )
    else:
        progress.info(f"FAILED: {error}")

    return {
        "briefing": parsed,
        "audit": audit,
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
        "status": status,
        "error": error,
        "audit_path": audit_path,
    }


def build_briefing_messages(page: dict[str, Any]) -> list[dict[str, str]]:
    candidate_text = page.get("text") or ""
    return [
        {
            "role": "system",
            "content": (
                "You read one page of the Haymarket trial transcript and produce a concise briefing "
                "that downstream stages will use as canonical context. Stay factual and only include "
                "information clearly supported by the page."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source ID: {page['id']}\n"
                f"URL: {page['url']}\n"
                f"Title: {page['title']}\n"
                f"Type: {page['source_type']}\n\n"
                "Return a JSON briefing with: source_id, summary (2-4 sentences), brief_title "
                "(short display title), navigation_summary (1-2 sentences for a source card), "
                "document_date, document_order, document_role, witness (name/role or null), "
                "examiners (name + side e.g. prosecution/defense), defendants_referenced, "
                "key_locations, key_dates, concise facetable topics, primary_people, "
                "primary_locations, referenced_events, document_events, and speaker_directory. "
                "document_date.normalized_date should be ISO YYYY-MM-DD when the source creation, "
                "testimony, exhibit, or procedural date is clear; otherwise null with original_text "
                "preserved. document_order should capture trial volume and page range when present. "
                "document_role must be one of testimony, exhibit, procedural, toc, cover, "
                "legal_document, other. primary_people and primary_locations are the most useful "
                "navigation entities for this document; set canonical_id only when the page itself "
                "clearly supports a stable person_/location_ id, otherwise null. referenced_events "
                "are historical events discussed by this document. Prefer concrete meetings, speeches, "
                "arrests, searches, marches, violence, police actions, trials, and other happenings "
                "described by the source. document_events are document/procedural lifecycle events "
                "that may still matter for navigation: publication, filing, court procedure, "
                "evidence introduction, document creation, or source description. Put filing, "
                "publication, cataloging, page copying, and introduction-into-evidence events in "
                "document_events, not referenced_events, unless the act is also a substantive "
                "historical event discussed by the page. Include event_time, place, participants, "
                "a short summary, page_refs, and a short source-grounded supporting_quote when "
                "available. page_refs should be the nearest page marker labels only (for example "
                "\"24\" or \"3 1/2\"), not ranges like \"pp. 24-27\". Set referenced_events "
                "canonical_id only when the page clearly reuses an existing event_ id, otherwise null. "
                "speaker_directory is a canonical map of every named speaker on this page (witness, "
                "examiners, judge, defendants, etc.). Each entry has a speaker_id, a display_name, "
                "and a role. The pipeline normalizes speaker_id to a canonical person_<slug> form "
                "based on display_name, so the speaker_id you provide is mostly informational — what "
                "matters is the display_name being accurate (e.g. 'John Bonfield' or 'Mr. Grinnell'). "
                "Downstream stages will use the canonical form as both the TEI <sp who=\"#…\"> "
                "reference and the entity id when this speaker is extracted as a person.\n\n"
                f"PAGE METADATA:\n{json.dumps(page.get('transcript_metadata', {}), ensure_ascii=False)}\n\n"
                f"CANDIDATE PAGE TEXT:\n{candidate_text}"
            ),
        },
    ]


def load_briefing_schema() -> dict[str, Any]:
    return load_schema_with_refs("briefing.schema.json")


def briefing_log_label(page: dict[str, Any], briefing: dict[str, Any] | None) -> str:
    briefing = briefing or {}
    role = str(briefing.get("document_role") or page.get("source_type") or "source").strip()
    witness = briefing.get("witness") if isinstance(briefing.get("witness"), dict) else {}
    witness_name = (witness or {}).get("name")
    if role == "testimony":
        return f"witness={witness_name or '—'}"

    title = str(briefing.get("brief_title") or page.get("title") or "").strip()
    title = compact_log_text(title)
    return f"{role}={title or page.get('id') or '—'}"


def compact_log_text(value: str, max_length: int = 72) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def normalize_speaker_directory(briefing: dict[str, Any] | None) -> None:
    """Rewrite speaker_directory IDs into canonical person_<slug> form.

    Stage A models tend to invent inconsistent ID styles ('#bonfield',
    '#mr_grinnell', 'bonfield', 'BONFIELD'), and Stage C person records
    must follow ^person_ per the schema. By overwriting the model's id
    with slugify(display_name, 'person'), every stage uses the same
    vocabulary: TEI <sp who="#person_john_bonfield">, person.id in
    Stage C, harmonized people list. No translation step needed.
    """
    if not briefing:
        return
    directory = briefing.get("speaker_directory")
    if not isinstance(directory, list):
        return
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in directory:
        if not isinstance(entry, dict):
            continue
        display_name = entry.get("display_name")
        if not display_name:
            continue
        canonical = slugify(str(display_name), "person")
        entry["speaker_id"] = canonical
        if canonical in seen:
            # Two model entries collapsed to the same canonical id — drop
            # the duplicate so downstream stages see one row per person.
            continue
        seen.add(canonical)
        deduped.append(entry)
    briefing["speaker_directory"] = deduped
