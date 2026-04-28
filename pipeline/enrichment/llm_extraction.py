from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from enrichment.progress import StageProgress
from enrichment.stage_briefing import run_briefing
from enrichment.stage_tagging import DEFAULT_MAX_WORKERS, run_tagging
from enrichment.stage_transcription import run_transcription
from utils.openai_schema import LLMCallError, MODEL_PRICING_PER_1M, estimate_cost_usd


DEFAULT_BRIEFING_MODEL = "gpt-4.1-mini"


def extract_pages_with_audit(
    pages: list[dict[str, Any]],
    storage,
    run_id: str,
    provider: str,
    models: list[str],
    briefing_model: str = DEFAULT_BRIEFING_MODEL,
    max_tagging_workers: int = DEFAULT_MAX_WORKERS,
    streaming: bool = True,
) -> dict[str, Any]:
    if provider != "openai":
        raise ValueError(f"Unsupported LLM provider: {provider}")

    audit_records: list[dict[str, Any]] = []
    bundles_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in models}

    briefings: dict[str, dict[str, Any]] = {}
    briefing_results: dict[str, dict[str, Any]] = {}

    extraction_pages = [page for page in pages if page.get("source_type") != "toc"]
    skipped = [page for page in pages if page.get("source_type") == "toc"]
    for page in skipped:
        print(f"Skipping LLM extraction for {page['id']} (source_type=toc)")

    for page in extraction_pages:
        progress = StageProgress(page["id"], "briefing", enabled=streaming)
        result = run_briefing(
            page=page,
            storage=storage,
            run_id=run_id,
            model=briefing_model,
            progress=progress,
        )
        briefing_results[page["id"]] = result
        if result["status"] == "success":
            briefings[page["id"]] = result["briefing"] or {}

    for model in models:
        for page in extraction_pages:
            briefing = briefings.get(page["id"])
            briefing_result = briefing_results.get(page["id"], _empty_briefing_result())

            transcription_progress = StageProgress(page["id"], "transcription", enabled=streaming)
            transcription = run_transcription(
                page=page,
                briefing=briefing,
                storage=storage,
                run_id=run_id,
                model=model,
                progress=transcription_progress,
            )

            tagging: dict[str, Any] | None = None
            bundle: dict[str, Any] | None = None
            if transcription["status"] == "success" and transcription.get("tei_xml"):
                tagging_progress = StageProgress(page["id"], "tagging", enabled=streaming)
                tagging = run_tagging(
                    page=page,
                    briefing=briefing,
                    tei_xml=transcription["tei_xml"],
                    storage=storage,
                    run_id=run_id,
                    model=model,
                    max_workers=max_tagging_workers,
                    progress=tagging_progress,
                )
                bundle = tagging["bundle"]
                bundle["tei_validation"] = transcription.get("validation")
                bundles_by_model[model].append(bundle)

            audit_records.append(
                build_combined_audit_record(
                    page=page,
                    model=model,
                    briefing_result=briefing_result,
                    transcription=transcription,
                    tagging=tagging,
                    bundle=bundle,
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


def _empty_briefing_result() -> dict[str, Any]:
    return {
        "briefing": None,
        "audit": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
        "status": "skipped",
        "error": None,
        "audit_path": None,
    }


def build_combined_audit_record(
    page: dict[str, Any],
    model: str,
    briefing_result: dict[str, Any],
    transcription: dict[str, Any],
    tagging: dict[str, Any] | None,
    bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_costs = {
        "briefing": float(briefing_result.get("cost_usd") or 0.0),
        "transcription": float(transcription.get("cost_usd") or 0.0),
        "tagging": float((tagging or {}).get("cost_usd") or 0.0),
    }
    stage_usage = {
        "briefing": briefing_result.get("usage") or _zero_usage(),
        "transcription": transcription.get("usage") or _zero_usage(),
        "tagging": (tagging or {}).get("usage") or _zero_usage(),
    }
    aggregate_usage = _zero_usage()
    for usage in stage_usage.values():
        aggregate_usage["input_tokens"] += usage.get("input_tokens", 0)
        aggregate_usage["output_tokens"] += usage.get("output_tokens", 0)
        aggregate_usage["total_tokens"] += usage.get("total_tokens", 0)

    if transcription["status"] != "success":
        status = "error"
        error = f"transcription: {transcription.get('error')}"
    elif tagging is None:
        status = "error"
        error = "tagging stage skipped"
    elif tagging.get("error_count", 0) > 0 and tagging.get("success_count", 0) == 0:
        status = "error"
        error = f"all {tagging['unit_count']} tagging units failed"
    else:
        status = "success"
        error = None

    return {
        "run_id": briefing_result.get("audit", {}).get("run_id") if briefing_result.get("audit") else None,
        "call_id": f"{page['id']}_{_slug(model)}",
        "page_id": page["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "input_diagnostics": transcription.get("diagnostics", {}),
        "stage_paths": {
            "briefing": briefing_result.get("audit_path"),
            "transcription": transcription.get("audit_path"),
            "tagging": (tagging or {}).get("audit_path"),
        },
        "stage_costs": {key: round(value, 8) for key, value in stage_costs.items()},
        "stage_usage": stage_usage,
        "tei_validation": (bundle or {}).get("tei_validation"),
        "tagging_summary": _summarize_tagging(tagging),
        "source_urls": [page["url"]],
        "usage": aggregate_usage,
        "cost_usd": round(sum(stage_costs.values()), 8),
        "status": status,
        "error": error,
    }


def _summarize_tagging(tagging: dict[str, Any] | None) -> dict[str, Any]:
    if not tagging:
        return {"unit_count": 0, "success_count": 0, "error_count": 0}
    return {
        "unit_count": tagging.get("unit_count", 0),
        "success_count": tagging.get("success_count", 0),
        "error_count": tagging.get("error_count", 0),
        "duration_s": tagging.get("duration_s", 0.0),
    }


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _slug(value: str) -> str:
    from utils.ids import slugify

    return slugify(value)


def build_cost_summary(run_id: str, audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _zero_cost_bucket()
    by_model: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, dict[str, Any]] = {
        "briefing": _zero_cost_bucket(),
        "transcription": _zero_cost_bucket(),
        "tagging": _zero_cost_bucket(),
    }
    calls = []

    for record in audit_records:
        usage = record["usage"]
        model = record["model"]
        totals["calls"] += 1
        totals["input_tokens"] += usage["input_tokens"]
        totals["output_tokens"] += usage["output_tokens"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["cost_usd"] += record["cost_usd"]

        model_totals = by_model.setdefault(model, _zero_cost_bucket())
        model_totals["calls"] += 1
        model_totals["input_tokens"] += usage["input_tokens"]
        model_totals["output_tokens"] += usage["output_tokens"]
        model_totals["total_tokens"] += usage["total_tokens"]
        model_totals["cost_usd"] += record["cost_usd"]

        stage_costs = record.get("stage_costs", {})
        stage_usage = record.get("stage_usage", {})
        for stage, bucket in by_stage.items():
            stage_cost = float(stage_costs.get(stage) or 0.0)
            stage_use = stage_usage.get(stage) or _zero_usage()
            if stage_cost == 0 and stage_use["total_tokens"] == 0:
                continue
            bucket["calls"] += 1
            bucket["input_tokens"] += stage_use["input_tokens"]
            bucket["output_tokens"] += stage_use["output_tokens"]
            bucket["total_tokens"] += stage_use["total_tokens"]
            bucket["cost_usd"] += stage_cost

        calls.append(
            {
                "call_id": record.get("call_id"),
                "provider": record.get("provider"),
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


def _zero_cost_bucket() -> dict[str, Any]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}


def build_model_eval(
    run_id: str,
    bundles_by_model: dict[str, list[dict[str, Any]]],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    models = []
    for model, bundles in bundles_by_model.items():
        records = [record for record in audit_records if record["model"] == model]
        successful = [record for record in records if record["status"] == "success"]
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
                "event_suggestion_count": sum(len(bundle.get("event_suggestions", [])) for bundle in bundles),
                "quote_count": sum(len(bundle.get("quotes", [])) for bundle in bundles),
                "tei_valid_count": sum(
                    1 for bundle in bundles if (bundle.get("tei_validation") or {}).get("status") == "valid"
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


__all__ = [
    "DEFAULT_BRIEFING_MODEL",
    "LLMCallError",
    "MODEL_PRICING_PER_1M",
    "build_cost_summary",
    "build_model_eval",
    "collect_missing_required_fields",
    "estimate_cost_usd",
    "extract_pages_with_audit",
]
