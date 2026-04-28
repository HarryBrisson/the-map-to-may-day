"""Three-stage LLM extraction orchestrator.

Stages:
- A) briefing       — once per page, fixed model (default gpt-4.1-mini), shared
                      across the per-model slate.
- B) transcription  — once per page per model. Produces validated TEI XML.
- C) tagging        — N concurrent calls per page per model, one per <sp> turn
                      (or per <p> on non-testimony pages). Produces the entity
                      bundle that downstream harmonization expects.

Public entry point: ``extract_pages_with_audit``.

Backwards-compat shims at the bottom (``call_openai_structured``,
``OPENAI_UNSUPPORTED_SCHEMA_KEYS``, ``load_openai_extraction_schema``,
``build_source_chunks``, ``build_messages``) so existing tests / callers keep
working while we migrate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from enrichment.progress import StageProgress
from enrichment.stage_briefing import run_briefing
from enrichment.stage_tagging import run_tagging
from enrichment.stage_transcription import (
    build_source_chunks as _stage_b_build_source_chunks,
    build_transcription_messages as _stage_b_build_messages,
    run_transcription,
)
from utils.openai_schema import (
    LLMCallError,  # noqa: F401  (re-exported for tests)
    MODEL_PRICING_PER_1M,
    OPENAI_UNSUPPORTED_SCHEMA_KEYS,
    estimate_cost_usd,
    load_extraction_bundle_schema as load_openai_extraction_schema,
)


PROMPT_TEMPLATE = "haymarket_three_stage_v1"
BRIEFING_MODEL_DEFAULT = "gpt-4.1-mini"


def extract_pages_with_audit(
    pages: list[dict[str, Any]],
    storage,
    run_id: str,
    provider: str,
    models: list[str],
    briefing_model: str = BRIEFING_MODEL_DEFAULT,
    max_tagging_workers: int = 8,
    streaming: bool = True,
) -> dict[str, Any]:
    if provider != "openai":
        raise ValueError(f"Unsupported LLM provider: {provider}")

    audit_records: list[dict[str, Any]] = []
    bundles_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in models}

    # ---- Stage A: briefing once per page, shared across models. ------------
    briefings: dict[str, dict[str, Any]] = {}
    briefing_audits: dict[str, dict[str, Any]] = {}
    for page in pages:
        if page.get("source_type") == "toc":
            print(f"Skipping LLM extraction for {page['id']} (source_type=toc)")
            continue

        progress = StageProgress(page["id"], stage="briefing", enabled=streaming)
        result = run_briefing(
            page=page,
            storage=storage,
            run_id=run_id,
            model=briefing_model,
            progress=progress,
        )
        if result["status"] != "success" or result["briefing"] is None:
            audit_records.append(
                _briefing_failure_record(page, briefing_model, run_id, result)
            )
            continue
        briefings[page["id"]] = result["briefing"]
        briefing_audits[page["id"]] = result

    # ---- Stages B + C per model -------------------------------------------
    for model in models:
        for page in pages:
            if page["id"] not in briefings:
                continue

            raw_html = (
                storage.read_text(page["raw_html_path"])
                if page.get("raw_html_path") and storage.exists(page["raw_html_path"])
                else ""
            )

            transcription_progress = StageProgress(
                page["id"], stage="transcription", enabled=streaming
            )
            transcription = run_transcription(
                page=page,
                briefing=briefings[page["id"]],
                storage=storage,
                run_id=run_id,
                model=model,
                raw_html=raw_html,
                progress=transcription_progress,
            )

            if transcription["status"] != "success" or not transcription["tei_xml"]:
                audit_records.append(
                    _combined_audit_record(
                        page=page,
                        model=model,
                        run_id=run_id,
                        briefing_result=briefing_audits[page["id"]],
                        transcription_result=transcription,
                        tagging_result=None,
                    )
                )
                continue

            tagging_progress = StageProgress(
                page["id"], stage="tagging", enabled=streaming
            )
            tagging = run_tagging(
                page=page,
                briefing=briefings[page["id"]],
                tei_xml=transcription["tei_xml"],
                storage=storage,
                run_id=run_id,
                model=model,
                max_workers=max_tagging_workers,
                progress=tagging_progress,
            )

            bundle = tagging["bundle"]
            # carry validation through for the model_eval rollup
            bundle["tei_validation"] = transcription["validation"]
            bundles_by_model[model].append(bundle)

            audit_records.append(
                _combined_audit_record(
                    page=page,
                    model=model,
                    run_id=run_id,
                    briefing_result=briefing_audits[page["id"]],
                    transcription_result=transcription,
                    tagging_result=tagging,
                )
            )

    cost_summary = build_cost_summary(run_id, audit_records)
    model_eval = build_model_eval(run_id, bundles_by_model, audit_records)
    storage.write_json(f"enriched/haymarket/llm_costs/{run_id}.json", cost_summary)
    storage.write_json(f"enriched/haymarket/model_evals/{run_id}.json", model_eval)

    return {
        "bundles_by_model": bundles_by_model,
        "audit_records": audit_records,
        "cost_summary": cost_summary,
        "model_eval": model_eval,
    }


# ----------------------------------------------------------------------------
# Audit record builders
# ----------------------------------------------------------------------------


def _briefing_failure_record(
    page: dict[str, Any],
    model: str,
    run_id: str,
    briefing_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "call_id": f"{page['id']}_briefing",
        "page_id": page["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": PROMPT_TEMPLATE,
        "stages": {"briefing": briefing_result["audit"]},
        "usage": briefing_result["usage"],
        "cost_usd": briefing_result["cost_usd"],
        "by_stage": {
            "briefing": {
                "usage": briefing_result["usage"],
                "cost_usd": briefing_result["cost_usd"],
            }
        },
        "parsed_output": None,
        "raw_output": None,
        "source_urls": [page.get("url")],
        "status": "error",
        "error": briefing_result.get("error") or "briefing failed",
    }


def _combined_audit_record(
    page: dict[str, Any],
    model: str,
    run_id: str,
    briefing_result: dict[str, Any],
    transcription_result: dict[str, Any],
    tagging_result: dict[str, Any] | None,
) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {
        "briefing": {
            "usage": briefing_result["usage"],
            "cost_usd": briefing_result["cost_usd"],
            "status": briefing_result["status"],
        },
        "transcription": {
            "usage": transcription_result["usage"],
            "cost_usd": transcription_result["cost_usd"],
            "status": transcription_result["status"],
        },
    }
    if tagging_result is not None:
        by_stage["tagging"] = {
            "usage": tagging_result["usage"],
            "cost_usd": tagging_result["cost_usd"],
            "status": tagging_result["status"],
            "unit_count": tagging_result.get("unit_count", 0),
        }

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    for stage in by_stage.values():
        for key in total_usage:
            total_usage[key] += stage["usage"].get(key, 0)
        total_cost += stage.get("cost_usd", 0.0)

    if tagging_result is not None and tagging_result["bundle"]:
        parsed_output = tagging_result["bundle"]
        status = "success"
        error = None
    elif transcription_result["status"] != "success":
        parsed_output = None
        status = "error"
        error = transcription_result.get("error") or "transcription failed"
    else:
        parsed_output = None
        status = "error"
        error = "tagging produced no bundle"

    return {
        "run_id": run_id,
        "call_id": f"{page['id']}_{model.replace('.', '_').replace('-', '_')}",
        "page_id": page["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "prompt_template": PROMPT_TEMPLATE,
        "stages": {
            "briefing": briefing_result["audit"],
            "transcription": transcription_result["audit"],
            "tagging_summary": (
                {
                    "unit_count": tagging_result.get("unit_count", 0),
                    "duration_s": tagging_result.get("duration_s"),
                    "audit_path": (
                        f"raw/haymarket/llm/{run_id}/"
                        f"{model.replace('.', '_').replace('-', '_')}/"
                        f"{page['id']}/tagging.jsonl"
                    ),
                }
                if tagging_result is not None
                else None
            ),
        },
        "by_stage": by_stage,
        "usage": total_usage,
        "cost_usd": round(total_cost, 8),
        "parsed_output": parsed_output,
        "raw_output": (
            transcription_result.get("audit", {}).get("raw_output")
            if not parsed_output
            else None
        ),
        "input_diagnostics": transcription_result.get("audit", {}).get(
            "input_diagnostics", {}
        ),
        "source_urls": [page.get("url")],
        "status": status,
        "error": error,
    }


# ----------------------------------------------------------------------------
# Cost / model-eval summaries
# ----------------------------------------------------------------------------


def build_cost_summary(run_id: str, audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    by_model: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, dict[str, Any]] = {}
    calls = []

    for record in audit_records:
        usage = record["usage"]
        model = record["model"]
        totals["calls"] += 1
        totals["input_tokens"] += usage["input_tokens"]
        totals["output_tokens"] += usage["output_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["cost_usd"] += record["cost_usd"]

        model_totals = by_model.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
        )
        model_totals["calls"] += 1
        model_totals["input_tokens"] += usage["input_tokens"]
        model_totals["output_tokens"] += usage["output_tokens"]
        model_totals["total_tokens"] += usage["total_tokens"]
        model_totals["cost_usd"] += record["cost_usd"]

        for stage_name, stage_info in (record.get("by_stage") or {}).items():
            stage_totals = by_stage.setdefault(
                stage_name,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
            )
            stage_usage = stage_info.get("usage", {})
            stage_totals["calls"] += 1
            stage_totals["input_tokens"] += stage_usage.get("input_tokens", 0)
            stage_totals["output_tokens"] += stage_usage.get("output_tokens", 0)
            stage_totals["total_tokens"] += stage_usage.get("total_tokens", 0)
            stage_totals["cost_usd"] += stage_info.get("cost_usd", 0.0)

        calls.append(
            {
                "call_id": record["call_id"],
                "provider": record["provider"],
                "model": model,
                "cost_usd": record["cost_usd"],
                "status": record["status"],
            }
        )

    totals["cost_usd"] = round(totals["cost_usd"], 8)
    for values in by_model.values():
        values["cost_usd"] = round(values["cost_usd"], 8)
    for values in by_stage.values():
        values["cost_usd"] = round(values["cost_usd"], 8)

    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "by_model": by_model,
        "by_stage": by_stage,
        "calls": calls,
    }


def build_model_eval(
    run_id: str,
    bundles_by_model: dict[str, list[dict[str, Any]]],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    models = []
    for model, bundles in bundles_by_model.items():
        records = [record for record in audit_records if record["model"] == model]
        successful = [record for record in records if record["parsed_output"]]
        cost = round(sum(record["cost_usd"] for record in records), 8)
        missing = collect_missing_required_fields(bundles)
        models.append(
            {
                "model": model,
                "pages": len(records),
                "parse_success_rate": round(len(successful) / len(records), 4) if records else 0,
                "schema_error_count": 0,
                "people_count": sum(len(bundle.get("people", [])) for bundle in bundles),
                "location_count": sum(len(bundle.get("locations", [])) for bundle in bundles),
                "claim_count": sum(len(bundle.get("claims", [])) for bundle in bundles),
                "event_suggestion_count": sum(
                    len(bundle.get("event_suggestions", [])) for bundle in bundles
                ),
                "quote_count": sum(len(bundle.get("quotes", [])) for bundle in bundles),
                "tei_valid_count": sum(
                    1
                    for bundle in bundles
                    if (bundle.get("tei_validation") or {}).get("status") == "valid"
                ),
                "missing_required_fields": sorted(missing),
                "cost_usd": cost,
            }
        )
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }


def collect_missing_required_fields(bundles: list[dict[str, Any]]) -> set[str]:
    required = {
        "bundle": ["tei_xml"],
        "people": ["id", "display_name", "roles", "bio"],
        "locations": ["id", "name", "address_1886", "coordinates"],
        "claims": ["id", "reported_by_person_id", "statement", "event_time"],
        "event_suggestions": ["id", "title", "time", "claim_ids"],
        "quotes": ["id", "speaker_person_id", "quote"],
    }
    missing: set[str] = set()
    for bundle in bundles:
        for field in required["bundle"]:
            if field not in bundle:
                missing.add(f"bundle.{field}")
        for collection, fields in required.items():
            if collection == "bundle":
                continue
            for item in bundle.get(collection, []):
                for field in fields:
                    if field not in item:
                        missing.add(f"{collection}.{field}")
    return missing


# ----------------------------------------------------------------------------
# Backwards-compat shims (kept so existing tests / external callers keep
# importing from ``enrichment.llm_extraction``).
# ----------------------------------------------------------------------------


def build_source_chunks(page: dict[str, Any], raw_html: str) -> list[dict[str, Any]]:
    return _stage_b_build_source_chunks(page, raw_html)


def build_messages(
    page: dict[str, Any], chunk: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    chunk = chunk or _stage_b_build_source_chunks(page, "")[0]
    return _stage_b_build_messages(page, chunk, briefing=None)


def call_openai_structured(model, input_messages):
    """Legacy single-shot helper preserved for tests that monkeypatch it.

    Returns the same (parsed, raw, usage) shape but uses the unified
    extraction-bundle schema to keep semantics identical to the prior path.
    """
    from utils.openai_schema import call_openai_structured as _call

    schema = load_openai_extraction_schema()
    return _call(
        model=model,
        input_messages=input_messages,
        schema=schema,
        schema_name="haymarket_extraction_bundle",
    )


__all__ = [
    "BRIEFING_MODEL_DEFAULT",
    "MODEL_PRICING_PER_1M",
    "OPENAI_UNSUPPORTED_SCHEMA_KEYS",
    "PROMPT_TEMPLATE",
    "build_cost_summary",
    "build_messages",
    "build_model_eval",
    "build_source_chunks",
    "call_openai_structured",
    "collect_missing_required_fields",
    "estimate_cost_usd",
    "extract_pages_with_audit",
    "load_openai_extraction_schema",
]
