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


BRIEFING_PROMPT_TEMPLATE = "haymarket_page_briefing_v1"
BRIEFING_MAX_OUTPUT_TOKENS = 2_000


def run_briefing(
    page: dict[str, Any],
    storage: JsonStorage,
    run_id: str,
    model: str = "gpt-4.1-mini",
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
        )
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
        witness = (parsed or {}).get("witness") or {}
        witness_name = witness.get("name") or "—"
        progress.done(
            f"witness={witness_name}, ${cost_usd:.4f}",
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
                "Return a JSON briefing with: source_id, summary (2-4 sentences), witness "
                "(name/role or null), examiners (name + side e.g. prosecution/defense), "
                "defendants_referenced, key_locations, key_dates, topics, and speaker_directory. "
                "speaker_directory is a canonical map of every named speaker on this page (witness, "
                "examiners, judge, defendants, etc.). Each entry has a speaker_id (kebab-case prefixed "
                "with #, e.g. '#bonfield', '#mr_grinnell'), a display_name, and a role. Downstream "
                "stages will use these IDs verbatim in TEI <sp who=\"...\"> attributes, so make them "
                "stable and short.\n\n"
                f"PAGE METADATA:\n{json.dumps(page.get('transcript_metadata', {}), ensure_ascii=False)}\n\n"
                f"CANDIDATE PAGE TEXT:\n{candidate_text}"
            ),
        },
    ]


def load_briefing_schema() -> dict[str, Any]:
    return load_schema_with_refs("briefing.schema.json")
