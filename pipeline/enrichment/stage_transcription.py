from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from enrichment.progress import StageProgress
from enrichment.tei_validation import (
    TEIValidationError,
    validate_generated_tei_or_raise,
)
from sources.hadc_source import extract_page_sections, serialize_xml, tei_tag
from utils.ids import slugify
from utils.openai_schema import (
    LLMCallError,
    call_openai_structured,
    estimate_cost_usd,
    normalize_openai_schema,
)
from utils.s3_storage import JsonStorage


TRANSCRIPTION_PROMPT_TEMPLATE = "haymarket_tei_transcription_v1"
FULL_DOCUMENT_CHAR_LIMIT = 120_000
RAW_HTML_EXCERPT_CHARS = 8_000
TRANSCRIPTION_MAX_OUTPUT_TOKENS = 32_000


def _build_transcription_schema() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_id", "tei_xml"],
        "properties": {
            "source_id": {"type": "string"},
            "tei_xml": {"type": "string"},
        },
    }
    normalize_openai_schema(schema)
    return schema


TRANSCRIPTION_SCHEMA = _build_transcription_schema()


def run_transcription(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    storage: JsonStorage,
    run_id: str,
    model: str,
    raw_html: str = "",
    progress: StageProgress | None = None,
) -> dict[str, Any]:
    """Run Stage B transcription. Returns:
    {tei_xml, validation, audit, usage, cost_usd, status, error, chunks}.
    """
    if progress:
        progress.info(f"starting ({model})")

    started = time.perf_counter()
    chunks = build_source_chunks(page, raw_html)
    parsed_chunks: list[dict[str, Any]] = []
    raw_outputs: list[Any] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    status = "success"
    error: str | None = None
    tei_xml: str | None = None
    validation: dict[str, Any] | None = None

    try:
        for chunk in chunks:
            messages = build_transcription_messages(page, chunk, briefing)
            parsed_chunk, raw_chunk, chunk_usage = call_openai_structured(
                model=model,
                input_messages=messages,
                schema=TRANSCRIPTION_SCHEMA,
                schema_name="haymarket_tei_transcription",
                max_output_tokens=TRANSCRIPTION_MAX_OUTPUT_TOKENS,
            )
            parsed_chunks.append(parsed_chunk)
            raw_outputs.append(raw_chunk)
            usage["input_tokens"] += chunk_usage["input_tokens"]
            usage["output_tokens"] += chunk_usage["output_tokens"]
            usage["total_tokens"] += chunk_usage["total_tokens"]

        if not parsed_chunks:
            raise LLMCallError("Stage B produced no chunks")

        if len(parsed_chunks) == 1:
            tei_xml = parsed_chunks[0].get("tei_xml") or ""
        else:
            tei_xml = combine_tei_documents(
                [chunk["tei_xml"] for chunk in parsed_chunks if chunk.get("tei_xml")]
            )

        if not tei_xml:
            raise LLMCallError("Stage B did not return tei_xml")

        validation = validate_generated_tei_or_raise(page, tei_xml)
    except TEIValidationError as exc:
        validation = exc.validation
        status = "error"
        error = str(exc)
        tei_xml = None
    except LLMCallError as exc:
        status = "error"
        error = str(exc)
        if exc.usage:
            for key in usage:
                usage[key] = max(usage[key], exc.usage.get(key, 0))
        if not raw_outputs and exc.raw_output is not None:
            raw_outputs.append(exc.raw_output)
        tei_xml = None
    except Exception as exc:
        status = "error"
        error = str(exc)
        tei_xml = None

    cost_usd = round(estimate_cost_usd(model, usage), 8)
    duration = time.perf_counter() - started

    audit = {
        "run_id": run_id,
        "stage": "transcription",
        "page_id": page["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": TRANSCRIPTION_PROMPT_TEMPLATE,
        "input_diagnostics": build_input_diagnostics(page, chunks, raw_html),
        "raw_output": raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs,
        "parsed_output": (
            {"tei_xml": tei_xml, "tei_validation": validation} if tei_xml else None
        ),
        "validation": validation,
        "usage": usage,
        "cost_usd": cost_usd,
        "status": status,
        "error": error,
        "duration_s": round(duration, 3),
    }

    model_path = slugify(model)
    storage.write_json(
        f"raw/haymarket/llm/{run_id}/{model_path}/{page['id']}/transcription.json",
        audit,
    )

    if progress:
        if status == "success" and tei_xml:
            ratio = (validation or {}).get("text", {}).get("similarity_ratio")
            recall = (validation or {}).get("text", {}).get("token_recall")
            extra = [
                f"{len(tei_xml):,} chars",
                f"ratio={ratio}" if ratio is not None else "",
                f"token_recall={recall}" if recall is not None else "",
                f"${cost_usd:.4f}",
            ]
            progress.done(", ".join(part for part in extra if part), duration_s=duration)
        else:
            progress.info(f"error: {error}")

    return {
        "tei_xml": tei_xml,
        "validation": validation,
        "audit": audit,
        "usage": usage,
        "cost_usd": cost_usd,
        "status": status,
        "error": error,
        "chunks": chunks,
    }


def build_transcription_messages(
    page: dict[str, Any],
    chunk: dict[str, Any],
    briefing: dict[str, Any] | None,
) -> list[dict[str, str]]:
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
    briefing_payload = briefing or {}
    speaker_directory = briefing_payload.get("speaker_directory") or []
    return [
        {
            "role": "system",
            "content": (
                "You convert messy Haymarket trial source HTML/text into well-formed TEI XML. "
                "Preserve source wording exactly except for whitespace normalization. "
                "Use TEI body markup for page breaks (<pb n=\"…\"/>), speaker turns "
                "(<sp who=\"#speaker_id\"><speaker>…</speaker><p>…</p></sp>), and inline "
                "<seg> spans where helpful. Use the canonical speaker directory below to "
                "set who=\"#…\" attributes; do not invent new speaker IDs unless required. "
                "Return only {source_id, tei_xml}. Do not extract entities — that is a "
                "separate downstream stage."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source ID: {page['id']}\n"
                f"URL: {page['url']}\n"
                f"Title: {page['title']}\n"
                f"Type: {page['source_type']}\n\n"
                f"BRIEFING JSON:\n{json.dumps(briefing_payload, ensure_ascii=False)}\n\n"
                f"SPEAKER DIRECTORY:\n{json.dumps(speaker_directory, ensure_ascii=False)}\n\n"
                f"SOURCE STRUCTURE JSON:\n{json.dumps(source_structure, ensure_ascii=False)}\n\n"
                f"RAW HTML EXCERPT:\n{chunk.get('raw_html_excerpt', '')}\n\n"
                f"CANDIDATE SOURCE TEXT ({chunk['mode']}, pages {chunk.get('page_range') or 'all'}):\n"
                f"{chunk['text']}"
            ),
        },
    ]


def build_source_chunks(
    page: dict[str, Any],
    raw_html: str,
    max_chars: int = FULL_DOCUMENT_CHAR_LIMIT,
) -> list[dict[str, Any]]:
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
            chunks.append(_build_chunk(page, chunks, current, raw_html_excerpt, text))
            current = []
            current_chars = 0
        current.append(section)
        current_chars += len(section_text)
    if current:
        chunks.append(_build_chunk(page, chunks, current, raw_html_excerpt, text))
    return chunks


def _build_chunk(
    page: dict[str, Any],
    chunks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    raw_html_excerpt: str,
    full_text: str,
) -> dict[str, Any]:
    page_refs = [section.get("page_ref") for section in sections if section.get("page_ref")]
    cues = [
        cue for cue in page.get("page_cues", [])
        if not page_refs or cue.get("page_ref") in page_refs
    ]
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
    named_speakers = sorted(
        {match.group(1).strip() for match in _re_finditer_speaker(text)}
    )
    context["named_speakers"] = named_speakers[:25]
    return context


def _re_finditer_speaker(text: str):
    return re.finditer(
        r"^((?:MR|Mr|THE COURT|WITNESS|A JUROR)[^:\n]*):", text, flags=re.MULTILINE
    )


def build_input_diagnostics(
    page: dict[str, Any], chunks: list[dict[str, Any]], raw_html: str
) -> dict[str, Any]:
    sent_chars = sum(
        len(chunk["text"]) + len(chunk.get("raw_html_excerpt", "")) for chunk in chunks
    )
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
