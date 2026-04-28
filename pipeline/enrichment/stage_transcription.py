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
TRANSCRIPTION_MAX_OUTPUT_TOKENS = 32_000
FULL_DOCUMENT_CHAR_LIMIT = 120_000
RAW_HTML_EXCERPT_CHARS = 8_000


def run_transcription(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    storage: JsonStorage,
    run_id: str,
    model: str,
    progress: StageProgress | None = None,
) -> dict[str, Any]:
    progress = progress or StageProgress(page["id"], "transcription", enabled=False)
    progress.info(f"starting ({model})")
    start = time.monotonic()

    raw_html = storage.read_text(page["raw_html_path"]) if page.get("raw_html_path") else ""
    chunks = build_source_chunks(page, raw_html)
    schema = transcription_schema()

    parsed_chunks: list[dict[str, Any]] = []
    raw_outputs: list[Any] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_messages: list[dict[str, str]] = []
    status = "success"
    error: str | None = None
    validation: dict[str, Any] | None = None
    tei_xml: str | None = None

    try:
        for chunk in chunks:
            chunk_messages = build_transcription_messages(page, briefing, chunk)
            input_messages.extend(chunk_messages)
            parsed_chunk, raw_output, chunk_usage = call_openai_structured(
                model=model,
                input_messages=chunk_messages,
                schema=schema,
                schema_name="haymarket_tei_transcription",
                max_output_tokens=TRANSCRIPTION_MAX_OUTPUT_TOKENS,
            )
            parsed_chunks.append(parsed_chunk)
            raw_outputs.append(raw_output)
            usage["input_tokens"] += chunk_usage["input_tokens"]
            usage["output_tokens"] += chunk_usage["output_tokens"]
            usage["total_tokens"] += chunk_usage["total_tokens"]

        tei_xml = combine_tei_documents([chunk["tei_xml"] for chunk in parsed_chunks if chunk.get("tei_xml")])
        validation = validate_generated_tei_or_raise(page, tei_xml)
    except TEIValidationError as exc:
        validation = exc.validation
        status = "error"
        error = str(exc)
    except Exception as exc:
        if isinstance(exc, LLMCallError) and exc.usage:
            usage = exc.usage
        status = "error"
        error = str(exc)

    cost_usd = estimate_cost_usd(model, usage)
    duration = time.monotonic() - start
    raw_output_serialized = raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs

    diagnostics = build_input_diagnostics(page, chunks, raw_html)

    audit = {
        "run_id": run_id,
        "page_id": page["id"],
        "stage": "transcription",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": TRANSCRIPTION_PROMPT_TEMPLATE,
        "input_messages": input_messages,
        "input_diagnostics": diagnostics,
        "raw_output": raw_output_serialized,
        "parsed_output": {"tei_xml": tei_xml, "tei_validation": validation} if tei_xml else None,
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
        "status": status,
        "error": error,
        "duration_s": round(duration, 3),
    }

    audit_path = f"raw/haymarket/llm/{run_id}/{slugify(model)}/{page['id']}/transcription.json"
    storage.write_json(audit_path, audit)

    if status == "success" and validation:
        text_validation = validation.get("text", {}) if isinstance(validation, dict) else {}
        ratio = text_validation.get("similarity_ratio", "n/a")
        token_recall = text_validation.get("token_recall", "n/a")
        progress.done(
            f"{text_validation.get('tei_chars', 0)} chars, ratio={ratio}, token_recall={token_recall}, ${cost_usd:.4f}",
            duration_s=duration,
        )
    else:
        progress.info(f"FAILED: {error}")

    return {
        "tei_xml": tei_xml,
        "validation": validation,
        "audit": audit,
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
        "status": status,
        "error": error,
        "audit_path": audit_path,
        "diagnostics": diagnostics,
    }


def build_transcription_messages(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    chunk: dict[str, Any],
) -> list[dict[str, str]]:
    speaker_directory = (briefing or {}).get("speaker_directory") or []
    briefing_summary = {
        "summary": (briefing or {}).get("summary"),
        "witness": (briefing or {}).get("witness"),
        "examiners": (briefing or {}).get("examiners"),
        "speaker_directory": speaker_directory,
    }
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
    return [
        {
            "role": "system",
            "content": (
                "You convert messy Haymarket trial source HTML/text into TEI XML. Preserve source "
                "wording exactly except for whitespace normalization. Use TEI body markup for page "
                "breaks, speaker turns, and inline references. Do NOT extract entities, claims, or "
                "quotes — that happens in a later stage. Return only structured TEI."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source ID: {page['id']}\n"
                f"URL: {page['url']}\n"
                f"Title: {page['title']}\n"
                f"Type: {page['source_type']}\n\n"
                "Return a JSON object with keys: source_id, tei_xml.\n"
                "The tei_xml MUST be well-formed XML following these rules:\n"
                "- The root element MUST be exactly: <TEI xmlns=\"http://www.tei-c.org/ns/1.0\"> "
                "(uppercase TEI, namespace declared).\n"
                "- All empty elements MUST be self-closed: <pb n=\"17\"/>, <lb/>, <gap/>. Never write "
                "<br>, <pb> with no slash, etc.\n"
                "- Do NOT use any HTML tags (no <br>, <b>, <i>, <img>, <a>). Use TEI equivalents: "
                "<lb/> for line breaks, <hi rend=\"bold\">…</hi> for emphasis, "
                "<figure><graphic url=\"…\"/></figure> for images.\n"
                "- Quote attribute values with double quotes; escape &, <, > as &amp;, &lt;, &gt; "
                "in text content.\n"
                "- Wrap each speaker turn in <sp who=\"#speaker_id\">…</sp> using the IDs from the "
                "speaker_directory below. If a speaker is unfamiliar, omit who but still wrap in <sp>.\n"
                "- Use <pb n=\"…\" facs=\"…\"/> for every page cue. Use <p> for paragraphs.\n"
                "- Preserve every word of the source — no summarization, no omission.\n\n"
                f"BRIEFING (use the speaker_directory for who=\"#…\" refs):\n"
                f"{json.dumps(briefing_summary, ensure_ascii=False)}\n\n"
                f"SOURCE STRUCTURE JSON:\n{json.dumps(source_structure, ensure_ascii=False)}\n\n"
                f"RAW HTML EXCERPT:\n{chunk.get('raw_html_excerpt', '')}\n\n"
                f"CANDIDATE SOURCE TEXT ({chunk['mode']}, pages {chunk.get('page_range') or 'all'}):\n"
                f"{chunk['text']}"
            ),
        },
    ]


def transcription_schema() -> dict[str, Any]:
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


def build_source_chunks(page: dict[str, Any], raw_html: str, max_chars: int = FULL_DOCUMENT_CHAR_LIMIT) -> list[dict[str, Any]]:
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
            chunks.append(build_chunk(page, chunks, current, raw_html_excerpt, text))
            current = []
            current_chars = 0
        current.append(section)
        current_chars += len(section_text)
    if current:
        chunks.append(build_chunk(page, chunks, current, raw_html_excerpt, text))
    return chunks


def build_chunk(
    page: dict[str, Any],
    chunks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    raw_html_excerpt: str,
    full_text: str,
) -> dict[str, Any]:
    page_refs = [section.get("page_ref") for section in sections if section.get("page_ref")]
    cues = [cue for cue in page.get("page_cues", []) if not page_refs or cue.get("page_ref") in page_refs]
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
    named_speakers = sorted({match.group(1).strip() for match in re_finditer_speaker(text)})
    context["named_speakers"] = named_speakers[:25]
    return context


def re_finditer_speaker(text: str):
    return re.finditer(r"^((?:MR|Mr|THE COURT|WITNESS|A JUROR)[^:\n]*):", text, flags=re.MULTILINE)


def build_input_diagnostics(page: dict[str, Any], chunks: list[dict[str, Any]], raw_html: str) -> dict[str, Any]:
    sent_chars = sum(len(chunk["text"]) + len(chunk.get("raw_html_excerpt", "")) for chunk in chunks)
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
    if len(tei_documents) == 1:
        return tei_documents[0]
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
