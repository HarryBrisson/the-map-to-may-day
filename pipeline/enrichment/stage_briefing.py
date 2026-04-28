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
    load_briefing_schema,
)
from utils.s3_storage import JsonStorage


BRIEFING_PROMPT_TEMPLATE = "haymarket_briefing_v1"
BRIEFING_TEXT_LIMIT = 60_000  # plenty for a single briefing pass; truncate if larger
BRIEFING_MAX_OUTPUT_TOKENS = 2_000


def run_briefing(
    page: dict[str, Any],
    storage: JsonStorage,
    run_id: str,
    model: str = "gpt-4.1-mini",
    progress: StageProgress | None = None,
) -> dict[str, Any]:
    """Run Stage A briefing and persist audit JSON.

    Returns: {briefing, audit, usage, cost_usd, status, error}.
    """
    if progress:
        progress.info(f"starting ({model})")

    started = time.perf_counter()
    candidate_text = (page.get("text") or "")[:BRIEFING_TEXT_LIMIT]
    input_messages = build_briefing_messages(page, candidate_text)
    schema = load_briefing_schema()

    parsed: dict[str, Any] | None = None
    raw_output: Any = None
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    status = "success"
    error: str | None = None

    try:
        parsed, raw_output, usage = call_openai_structured(
            model=model,
            input_messages=input_messages,
            schema=schema,
            schema_name="haymarket_briefing",
            max_output_tokens=BRIEFING_MAX_OUTPUT_TOKENS,
        )
        # Guarantee required source_id even if model omits it.
        parsed.setdefault("source_id", page["id"])
    except LLMCallError as exc:
        raw_output = exc.raw_output
        usage = exc.usage
        status = "error"
        error = str(exc)
    except Exception as exc:  # network / SDK errors
        status = "error"
        error = str(exc)

    cost_usd = round(estimate_cost_usd(model, usage), 8)
    duration = time.perf_counter() - started

    audit = {
        "run_id": run_id,
        "stage": "briefing",
        "page_id": page["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": BRIEFING_PROMPT_TEMPLATE,
        "input_messages": input_messages,
        "raw_output": raw_output,
        "parsed_output": parsed,
        "usage": usage,
        "cost_usd": cost_usd,
        "status": status,
        "error": error,
        "duration_s": round(duration, 3),
    }

    model_path = slugify(model)
    storage.write_json(
        f"raw/haymarket/llm/{run_id}/{model_path}/{page['id']}/briefing.json",
        audit,
    )

    if progress:
        if status == "success":
            witness_name = (parsed or {}).get("witness", {}) or {}
            witness_label = witness_name.get("name") if isinstance(witness_name, dict) else None
            extra = []
            if witness_label:
                extra.append(f"witness={witness_label}")
            extra.append(f"${cost_usd:.4f}")
            progress.done(", ".join(extra), duration_s=duration)
        else:
            progress.info(f"error: {error}")

    return {
        "briefing": parsed,
        "audit": audit,
        "usage": usage,
        "cost_usd": cost_usd,
        "status": status,
        "error": error,
    }


def build_briefing_messages(page: dict[str, Any], candidate_text: str) -> list[dict[str, str]]:
    metadata = page.get("transcript_metadata", {}) or {}
    context = {
        "source_id": page["id"],
        "title": page.get("title"),
        "url": page.get("url"),
        "source_type": page.get("source_type"),
        "transcript_metadata": metadata,
        "page_cues": page.get("page_cues", []),
    }
    return [
        {
            "role": "system",
            "content": (
                "You produce a structured briefing for one Haymarket trial source page. "
                "Identify the witness (if any), the examiners, defendants who are referenced, "
                "key locations, key dates, recurring topics, and a canonical speaker directory "
                "that downstream tagging can use to resolve <sp who=\"#…\"> references. "
                "Speaker IDs must be lowercase snake-case, prefixed with 'speaker_' "
                "(e.g. speaker_bonfield, speaker_grinnell, speaker_court, speaker_witness). "
                "Only include information supported by the candidate text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PAGE CONTEXT JSON:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                f"CANDIDATE SOURCE TEXT:\n{candidate_text}"
            ),
        },
    ]
