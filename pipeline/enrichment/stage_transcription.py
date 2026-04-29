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
from sources.hadc_source import (
    XML_NS,
    extract_page_sections,
    parse_tei_xml,
    serialize_xml,
    tei_tag,
)
from utils.ids import slugify
from utils.openai_schema import (
    LLMCallError,
    call_openai_text,
    estimate_cost_usd,
)
from utils.s3_storage import JsonStorage


TRANSCRIPTION_PROMPT_TEMPLATE = "haymarket_tei_transcription_v1"
TRANSCRIPTION_JSONL_PROMPT_TEMPLATE = "haymarket_jsonl_transcription_v1"
TRANSCRIPTION_MAX_OUTPUT_TOKENS = 32_000
# TEI markup expands the output past the input, so the per-chunk input must
# leave room for tags inside the 32k output token budget. 50k input chars is
# roughly 12-13k tokens; expanded TEI output lands around 17-20k tokens, with
# safety margin under the 32k cap. Stay above ~42k so the i019_052 reference
# page (a known-good single-chunk transcription) does not get split. Going
# much lower (e.g. 32k) makes models summarize each small chunk independently
# and hurts recall.
FULL_DOCUMENT_CHAR_LIMIT = 50_000
RAW_HTML_EXCERPT_CHARS = 8_000
DEFAULT_TRANSCRIPTION_RETRIES = 2  # total attempts including the first


def run_transcription(
    page: dict[str, Any],
    briefing: dict[str, Any] | None,
    storage: JsonStorage,
    run_id: str,
    model: str,
    progress: StageProgress | None = None,
    max_attempts: int = DEFAULT_TRANSCRIPTION_RETRIES,
    output_format: str = "tei",
) -> dict[str, Any]:
    progress = progress or StageProgress(page["id"], "transcription", enabled=False)
    progress.info(f"starting ({model}, format={output_format})")
    start = time.monotonic()
    if output_format not in ("tei", "jsonl"):
        raise ValueError(f"Unsupported transcription format: {output_format}")

    raw_html = storage.read_text(page["raw_html_path"]) if page.get("raw_html_path") else ""
    chunks = build_source_chunks(page, raw_html)

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_messages: list[dict[str, str]] = []
    status = "error"
    error: str | None = None
    validation: dict[str, Any] | None = None
    tei_xml: str | None = None
    raw_outputs: list[Any] = []
    chunk_tei_documents: list[str] = []
    attempts = 0
    attempt_history: list[dict[str, Any]] = []

    all_jsonl_rows: list[dict[str, Any]] = []
    while attempts < max_attempts:
        attempts += 1
        attempt_chunk_tei: list[str] = []
        attempt_raw_outputs: list[Any] = []
        attempt_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        attempt_messages: list[dict[str, str]] = []
        attempt_jsonl_rows: list[dict[str, Any]] = []
        attempt_status = "success"
        attempt_error: str | None = None
        attempt_validation: dict[str, Any] | None = None
        attempt_tei: str | None = None
        try:
            for chunk in chunks:
                if output_format == "jsonl":
                    chunk_messages = build_jsonl_messages(page, briefing, chunk)
                else:
                    chunk_messages = build_transcription_messages(page, briefing, chunk)
                attempt_messages.extend(chunk_messages)
                raw_text, raw_output, chunk_usage = call_openai_text(
                    model=model,
                    input_messages=chunk_messages,
                    max_output_tokens=TRANSCRIPTION_MAX_OUTPUT_TOKENS,
                )
                if output_format == "jsonl":
                    rows = parse_jsonl_rows(raw_text)
                    if not rows:
                        raise LLMCallError(
                            "Model produced no parseable JSONL rows. First 200 chars: "
                            + (raw_text or "")[:200].replace("\n", " ")
                        )
                    attempt_jsonl_rows.extend(rows)
                else:
                    chunk_tei = extract_tei_from_text(raw_text)
                    attempt_chunk_tei.append(chunk_tei)
                attempt_raw_outputs.append(raw_output)
                attempt_usage["input_tokens"] += chunk_usage["input_tokens"]
                attempt_usage["output_tokens"] += chunk_usage["output_tokens"]
                attempt_usage["total_tokens"] += chunk_usage["total_tokens"]

            if output_format == "jsonl":
                attempt_tei = synthesize_tei_from_rows(page, attempt_jsonl_rows)
            else:
                attempt_tei = combine_tei_documents(attempt_chunk_tei)
            attempt_validation = validate_generated_tei_or_raise(page, attempt_tei)
        except TEIValidationError as exc:
            attempt_validation = exc.validation
            attempt_status = "error"
            attempt_error = str(exc)
        except Exception as exc:
            if isinstance(exc, LLMCallError) and exc.usage:
                attempt_usage = exc.usage
            attempt_status = "error"
            attempt_error = str(exc)

        usage["input_tokens"] += attempt_usage["input_tokens"]
        usage["output_tokens"] += attempt_usage["output_tokens"]
        usage["total_tokens"] += attempt_usage["total_tokens"]
        attempt_history.append(
            {
                "attempt": attempts,
                "status": attempt_status,
                "error": attempt_error,
                "usage": attempt_usage,
                "tei_chars": len(attempt_tei or ""),
                "validation": attempt_validation,
            }
        )

        # Always remember the last attempt's outputs for the audit; only adopt as
        # the final result if it succeeded.
        input_messages = attempt_messages
        chunk_tei_documents = attempt_chunk_tei
        all_jsonl_rows = attempt_jsonl_rows
        raw_outputs = attempt_raw_outputs
        validation = attempt_validation
        tei_xml = attempt_tei
        status = attempt_status
        error = attempt_error

        if attempt_status == "success":
            break

        if attempts < max_attempts:
            progress.info(f"attempt {attempts}/{max_attempts} failed ({attempt_error}); retrying")

    cost_usd = estimate_cost_usd(model, usage)
    duration = time.monotonic() - start
    raw_output_serialized = raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs

    diagnostics = build_input_diagnostics(page, chunks, raw_html)

    parsed_output: dict[str, Any] | None = None
    if tei_xml:
        parsed_output = {"tei_xml": tei_xml, "tei_validation": validation}
        if output_format == "jsonl":
            parsed_output["jsonl_rows"] = all_jsonl_rows
            parsed_output["jsonl_row_count"] = len(all_jsonl_rows)

    audit = {
        "run_id": run_id,
        "page_id": page["id"],
        "stage": "transcription",
        "format": output_format,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": (
            TRANSCRIPTION_JSONL_PROMPT_TEMPLATE
            if output_format == "jsonl"
            else TRANSCRIPTION_PROMPT_TEMPLATE
        ),
        "input_messages": input_messages,
        "input_diagnostics": diagnostics,
        "raw_output": raw_output_serialized,
        "parsed_output": parsed_output,
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
        "status": status,
        "error": error,
        "duration_s": round(duration, 3),
        "attempts": attempts,
        "attempt_history": [
            {
                "attempt": entry["attempt"],
                "status": entry["status"],
                "error": entry["error"],
                "usage": entry["usage"],
                "tei_chars": entry["tei_chars"],
            }
            for entry in attempt_history
        ],
    }

    audit_path = f"raw/haymarket/llm/{run_id}/{slugify(model)}/{page['id']}/transcription.json"
    storage.write_json(audit_path, audit)

    attempt_suffix = f", attempts={attempts}" if attempts > 1 else ""
    if status == "success" and validation:
        text_validation = validation.get("text", {}) if isinstance(validation, dict) else {}
        ratio = text_validation.get("similarity_ratio", "n/a")
        token_recall = text_validation.get("token_recall", "n/a")
        progress.done(
            f"{text_validation.get('tei_chars', 0)} chars, ratio={ratio}, "
            f"token_recall={token_recall}, ${cost_usd:.4f}{attempt_suffix}",
            duration_s=duration,
        )
    else:
        progress.info(f"FAILED: {error}")
        candidate_chars = len(page.get("text") or "")
        raw_chars = sum(_raw_output_text_length(raw) for raw in raw_outputs)
        merged_chars = len(tei_xml or "")
        progress.info(
            f"  output: raw={raw_chars} chars across {len(raw_outputs)} chunk(s), "
            f"merged_tei={merged_chars} chars, candidate={candidate_chars} chars"
        )
        if isinstance(validation, dict):
            text_validation = validation.get("text", {})
            if text_validation:
                progress.info(
                    f"  text validation: ratio={text_validation.get('similarity_ratio', '?')}, "
                    f"token_recall={text_validation.get('token_recall', '?')}"
                )
            page_marker_stats = validation.get("page_markers", {})
            if page_marker_stats:
                missing = page_marker_stats.get("missing", [])
                progress.info(
                    f"  page_markers: expected={page_marker_stats.get('expected')}, "
                    f"actual={page_marker_stats.get('actual')}, "
                    f"missing={len(missing)}"
                    + (f" first_missing={missing[:5]}" if missing else "")
                )
            for line in (validation.get("errors") or [])[:3]:
                progress.info(f"  reason: {line}")
        progress.info(f"  audit: {audit_path}")

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
                "You transcribe Haymarket trial source text into TEI XML, verbatim. Your output's "
                "text content (with tags stripped) must match the source word for word; you are not "
                "summarizing, paraphrasing, or omitting. Entity extraction happens in a later stage. "
                "Output only the TEI XML document — no markdown code fences, no prose preamble, no "
                "trailing commentary. Begin with <TEI ...> and end with </TEI>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source ID: {page['id']}\n"
                f"URL: {page['url']}\n"
                f"Title: {page['title']}\n"
                f"Type: {page['source_type']}\n\n"
                "Produce a TEI document for the CANDIDATE SOURCE TEXT below.\n\n"
                "EXAMPLE INPUT (candidate text):\n"
                "[Image, Volume I, Page 17]\n"
                "Q What is your name?\n"
                "A John Bonfield.\n"
                "[Image, Volume I, Page 18]\n"
                "Whereupon a recess was taken until 2 P.M.\n\n"
                "EXAMPLE OUTPUT (TEI):\n"
                "<TEI xmlns=\"http://www.tei-c.org/ns/1.0\">\n"
                "  <text>\n"
                "    <body>\n"
                "      <pb n=\"17\"/>\n"
                "      <sp who=\"#person_mr_grinnell\">\n"
                "        <speaker>Q.</speaker>\n"
                "        <p>What is your name?</p>\n"
                "      </sp>\n"
                "      <sp who=\"#person_john_bonfield\">\n"
                "        <speaker>A.</speaker>\n"
                "        <p>John Bonfield.</p>\n"
                "      </sp>\n"
                "      <pb n=\"18\"/>\n"
                "      <p>Whereupon a recess was taken until 2 P.M.</p>\n"
                "    </body>\n"
                "  </text>\n"
                "</TEI>\n\n"
                "INTERRUPTIONS — IMPORTANT.\n"
                "Trial testimony is often interrupted by objections, side comments, or the witness "
                "restating. When that happens, close the current <sp> and <p> before starting the "
                "next speaker. Speech turns NEVER nest — every open <sp> has a matching </sp> "
                "before another <sp> begins.\n\n"
                "EXAMPLE INPUT (interrupted answer):\n"
                "Q What did you see?\n"
                "A I saw a man in a dark coat--\n"
                "MR. SALOMON: Object, leading the witness.\n"
                "THE COURT: Overruled.\n"
                "A approaching the wagon.\n\n"
                "EXAMPLE OUTPUT:\n"
                "<sp who=\"#person_mr_grinnell\">\n"
                "  <speaker>Q</speaker>\n"
                "  <p>What did you see?</p>\n"
                "</sp>\n"
                "<sp who=\"#person_john_bonfield\">\n"
                "  <speaker>A</speaker>\n"
                "  <p>I saw a man in a dark coat--</p>\n"
                "</sp>\n"
                "<sp who=\"#person_mr_salomon\">\n"
                "  <speaker>Mr. SALOMON:</speaker>\n"
                "  <p>Object, leading the witness.</p>\n"
                "</sp>\n"
                "<sp who=\"#person_the_court\">\n"
                "  <speaker>THE COURT:</speaker>\n"
                "  <p>Overruled.</p>\n"
                "</sp>\n"
                "<sp who=\"#person_john_bonfield\">\n"
                "  <speaker>A</speaker>\n"
                "  <p>approaching the wagon.</p>\n"
                "</sp>\n\n"
                "Rules:\n"
                "- Root is <TEI xmlns=\"http://www.tei-c.org/ns/1.0\">; empty elements are self-closed.\n"
                "- Each [Image, Volume X, Page N] marker becomes a <pb n=\"N\"/> element.\n"
                "- Each speech turn (Q, A, or a named speaker like 'MR. GRINNELL: …') is wrapped in "
                "<sp who=\"#person_<slug>\">…</sp>, where the speaker_id comes EXACTLY from the "
                "speaker_directory in the briefing — copy them verbatim, do not invent variants. "
                "The <sp> contains a <speaker> with the literal label ('Q.', 'A.', 'MR. GRINNELL') "
                "and one or more <p> elements with the spoken text. If a speaker is not in the "
                "directory, drop the who attribute but still wrap in <sp>.\n"
                "- Plain narrative or stage directions (recesses, descriptions of exhibits) go in a "
                "bare <p>, not inside an <sp>.\n"
                "- Escape &, <, > in text content as &amp;, &lt;, &gt; (e.g. '&c.' becomes "
                "'&amp;c.').\n"
                "- Transcribe every word of the source verbatim, including the leading 'Q' or 'A' "
                "speaker label inside <speaker>. Long testimony stays long.\n"
                "- Entity extraction (people, locations, claims) happens in a later stage — do not "
                "add <seg> or <annotation> elements here.\n\n"
                f"PAGE BRIEFING (use speaker_directory for who=\"#…\" refs):\n"
                f"{json.dumps(briefing_summary, ensure_ascii=False)}\n\n"
                f"SOURCE STRUCTURE:\n{json.dumps(source_structure, ensure_ascii=False)}\n\n"
                f"RAW HTML EXCERPT:\n{chunk.get('raw_html_excerpt', '')}\n\n"
                f"CANDIDATE SOURCE TEXT ({chunk['mode']}, pages {chunk.get('page_range') or 'all'}):\n"
                f"{chunk['text']}"
            ),
        },
    ]


def _raw_output_text_length(raw_output: Any) -> int:
    if not raw_output:
        return 0
    if isinstance(raw_output, str):
        return len(raw_output)
    if isinstance(raw_output, dict):
        text = raw_output.get("output_text")
        if isinstance(text, str):
            return len(text)
        for item in raw_output.get("output") or []:
            for content in (item or {}).get("content") or []:
                if (content or {}).get("type") == "output_text":
                    return len(content.get("text") or "")
    return 0


def build_jsonl_messages(
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
        "page_cues": chunk.get("page_cues", []),
        "speaker_context": chunk.get("speaker_context", {}),
        "toc_entries": page.get("toc_entries", [])[:60],
    }
    return [
        {
            "role": "system",
            "content": (
                "You transcribe Haymarket trial source text into JSONL — one JSON object per line. "
                "Each line of the source becomes one JSON object on its own line, in source order. "
                "You are not summarizing or paraphrasing; the union of every row's 'text' field "
                "must contain every word of the source. Output only the JSONL — no markdown code "
                "fences, no prose preamble, no trailing commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source ID: {page['id']}\n"
                f"URL: {page['url']}\n"
                f"Title: {page['title']}\n"
                f"Type: {page['source_type']}\n\n"
                "Output JSONL — one JSON object per line, no surrounding array. Each object has "
                "these fields:\n"
                "- page: the most recent page number as a string (e.g. \"17\"), or null if none "
                "yet seen.\n"
                "- speaker: the literal speaker label as it appears in the source ('Q', 'A', "
                "'MR. GRINNELL', 'THE COURT', etc.), or null for narrative or stage directions.\n"
                "- speaker_id: a stable id from the briefing's speaker_directory, prefixed with "
                "'#' (e.g. '#person_john_bonfield', '#person_mr_grinnell'). null if speaker is null or unidentifiable.\n"
                "- text: the spoken or narrative text, verbatim from the source.\n\n"
                "EXAMPLE INPUT (candidate text):\n"
                "[Image, Volume I, Page 17]\n"
                "Q What is your name?\n"
                "A John Bonfield.\n"
                "[Image, Volume I, Page 18]\n"
                "Whereupon a recess was taken until 2 P.M.\n\n"
                "EXAMPLE OUTPUT (JSONL):\n"
                '{"page":"17","speaker":"Q","speaker_id":"#person_mr_grinnell","text":"What is your name?"}\n'
                '{"page":"17","speaker":"A","speaker_id":"#person_john_bonfield","text":"John Bonfield."}\n'
                '{"page":"18","speaker":null,"speaker_id":null,"text":"Whereupon a recess was taken until 2 P.M."}\n\n'
                "Rules:\n"
                "- Image markers like [Image, Volume X, Page N] do NOT get their own row. They "
                "update the running 'page' value, which appears on the next row.\n"
                "- An interrupted answer (e.g. witness, then objection, then witness resumes) "
                "becomes three separate rows — each row is one speaker's turn.\n"
                "- Long answers stay long — emit one row per speaker turn even if the text is "
                "thousands of characters.\n"
                "- Do not fabricate a speaker_id; if the speaker doesn't appear in the briefing's "
                "speaker_directory, set speaker_id to null.\n\n"
                f"PAGE BRIEFING (use speaker_directory for speaker_id values):\n"
                f"{json.dumps(briefing_summary, ensure_ascii=False)}\n\n"
                f"SOURCE STRUCTURE:\n{json.dumps(source_structure, ensure_ascii=False)}\n\n"
                f"CANDIDATE SOURCE TEXT ({chunk['mode']}, pages {chunk.get('page_range') or 'all'}):\n"
                f"{chunk['text']}"
            ),
        },
    ]


def parse_jsonl_rows(raw_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not raw_text:
        return rows
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown fences and stray prose lines that aren't JSON
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if "text" not in row:
            continue
        rows.append(row)
    return rows


def synthesize_tei_from_rows(page: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    root = ET.Element(tei_tag("TEI"), {f"{{{XML_NS}}}id": page["id"]})
    text_el = ET.SubElement(root, tei_tag("text"))
    body = ET.SubElement(text_el, tei_tag("body"))

    facs_by_page = {
        str(cue.get("page_ref")): cue.get("facs")
        for cue in page.get("page_cues") or []
        if cue.get("page_ref")
    }

    current_page: str | None = None
    sp_index = 0
    for row in rows:
        page_ref = row.get("page")
        page_str = str(page_ref) if page_ref not in (None, "") else None
        if page_str and page_str != current_page:
            attrs = {"n": page_str}
            facs = facs_by_page.get(page_str)
            if facs:
                attrs["facs"] = facs
            ET.SubElement(body, tei_tag("pb"), attrs)
            current_page = page_str

        body_text = str(row.get("text") or "").strip()
        if not body_text:
            continue

        speaker_label = row.get("speaker")
        speaker_id = row.get("speaker_id")
        if speaker_label:
            sp_attrs = {"n": str(sp_index)}
            if speaker_id:
                normalized_id = speaker_id if str(speaker_id).startswith("#") else f"#{speaker_id}"
                sp_attrs["who"] = normalized_id
            sp_el = ET.SubElement(body, tei_tag("sp"), sp_attrs)
            ET.SubElement(sp_el, tei_tag("speaker")).text = str(speaker_label)
            ET.SubElement(sp_el, tei_tag("p")).text = body_text
            sp_index += 1
        else:
            ET.SubElement(body, tei_tag("p")).text = body_text

    return serialize_xml(root)


def extract_tei_from_text(raw_text: str) -> str:
    """Pull the <TEI>…</TEI> document out of a plain-text completion.

    Tolerates models that wrap their output in markdown code fences
    (```xml … ```) or include a short prose preamble before the XML.
    """
    if not raw_text:
        raise LLMCallError("Model returned empty output for TEI transcription")
    start_match = re.search(r"<TEI\b", raw_text)
    if not start_match:
        raise LLMCallError(
            "Model did not return a <TEI> element. First 200 chars: "
            + raw_text[:200].replace("\n", " ")
        )
    end_match = re.search(r"</TEI\s*>", raw_text)
    if end_match is None:
        return raw_text[start_match.start():].rstrip()
    return raw_text[start_match.start():end_match.end()]


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
    first_root = parse_tei_xml(tei_documents[0])
    first_div = first_root.find(f".//{tei_tag('body')}/{tei_tag('div')}")
    if first_div is None:
        first_body = first_root.find(f".//{tei_tag('body')}")
        if first_body is None:
            return tei_documents[0]
        first_div = first_body

    first_standoff = first_root.find(tei_tag("standOff"))
    for tei_xml in tei_documents[1:]:
        root = parse_tei_xml(tei_xml)
        div = root.find(f".//{tei_tag('body')}/{tei_tag('div')}")
        body = root.find(f".//{tei_tag('body')}")
        source = div if div is not None else body
        if source is not None:
            for child in list(source):
                first_div.append(child)
        standoff = root.find(tei_tag("standOff"))
        if standoff is not None:
            if first_standoff is None:
                first_standoff = ET.SubElement(first_root, tei_tag("standOff"))
            for child in list(standoff):
                first_standoff.append(child)
    return serialize_xml(first_root)
