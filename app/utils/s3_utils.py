from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from utils.config import get_data_root, get_s3_bucket, get_s3_prefix, get_storage_backend


logger = logging.getLogger(__name__)

DATASET_PATHS = {
    "people": "enriched/haymarket/people/latest.json",
    "events": "enriched/haymarket/events/latest.json",
    "claims": "enriched/haymarket/claims/latest.json",
    "locations": "enriched/haymarket/locations/latest.json",
    "sources": "enriched/haymarket/sources/latest.json",
}

HADC_LATEST_RUN_PATH = "raw/haymarket/hadc/latest_run.json"


def _empty_dataset(dataset_name: str) -> list[dict[str, Any]]:
    logger.info("Dataset %s is not available yet; returning empty list", dataset_name)
    return []


def _read_local_json(relative_path: str) -> Any:
    data_root = get_data_root()
    path = (data_root / relative_path).resolve()
    if not _is_relative_to(path, data_root.resolve()):
        raise ValueError(f"Refusing to read outside data root: {relative_path}")

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_local_text(relative_path: str) -> str | None:
    data_root = get_data_root()
    path = (data_root / relative_path).resolve()
    if not _is_relative_to(path, data_root.resolve()):
        raise ValueError(f"Refusing to read outside data root: {relative_path}")

    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def _read_s3_json(relative_path: str) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on deploy environment
        raise RuntimeError("boto3 is required for S3 storage") from exc

    prefix = get_s3_prefix()
    key = f"{prefix}/{relative_path}" if prefix else relative_path
    client = boto3.client("s3")
    response = client.get_object(Bucket=get_s3_bucket(), Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def _read_s3_text(relative_path: str) -> str | None:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:  # pragma: no cover - depends on deploy environment
        raise RuntimeError("boto3 is required for S3 storage") from exc

    prefix = get_s3_prefix()
    key = f"{prefix}/{relative_path}" if prefix else relative_path
    client = boto3.client("s3")
    try:
        response = client.get_object(Bucket=get_s3_bucket(), Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None
        raise
    return response["Body"].read().decode("utf-8")


def read_json(relative_path: str) -> Any:
    backend = get_storage_backend()
    if backend == "s3":
        return _read_s3_json(relative_path)
    if backend == "local":
        return _read_local_json(relative_path)
    raise ValueError(f"Unsupported storage backend: {backend}")


def read_text(relative_path: str) -> str | None:
    backend = get_storage_backend()
    if backend == "s3":
        return _read_s3_text(relative_path)
    if backend == "local":
        return _read_local_text(relative_path)
    raise ValueError(f"Unsupported storage backend: {backend}")


def read_dataset(dataset_name: str) -> list[dict[str, Any]]:
    relative_path = DATASET_PATHS[dataset_name]
    data = read_json(relative_path)
    if data is None:
        return _empty_dataset(dataset_name)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(f"Dataset {dataset_name} must be a list or object with an items list")


def read_latest_hadc_run() -> dict[str, Any] | None:
    return read_json(HADC_LATEST_RUN_PATH)


def read_transcript_catalog() -> list[dict[str, Any]]:
    latest_run = read_latest_hadc_run()
    if not latest_run:
        return []
    pages = read_json(f"raw/haymarket/hadc/{latest_run['run_id']}/pages.json") or []
    source_summaries = {source.get("id"): source for source in read_dataset("sources")}
    return [
        with_navigation_fallback(
            {
                "id": page["id"],
                "url": page["url"],
                "title": page["title"],
                "source_type": page["source_type"],
                "transcript_metadata": page.get("transcript_metadata", {}),
                "source_stats": page.get("source_stats", {}),
                "tei_path": page.get("tei_path"),
                "transcript_json_path": page.get("transcript_json_path"),
            },
            source_summaries.get(page["id"]),
        )
        for page in pages
    ]


def with_navigation_fallback(page_summary: dict[str, Any], enriched_summary: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(page_summary)
    if enriched_summary:
        for key in ("navigation", "candidate_text_path", "tei_path", "transcript_json_path"):
            if enriched_summary.get(key) is not None:
                result[key] = enriched_summary[key]
    result["navigation"] = normalize_navigation(result, result.get("navigation"))
    return result


def normalize_navigation(source: dict[str, Any], navigation: Any) -> dict[str, Any]:
    metadata = source.get("transcript_metadata", {}) or {}
    nav = navigation if isinstance(navigation, dict) else {}
    document_date = nav.get("document_date") if isinstance(nav.get("document_date"), dict) else {}
    document_order = nav.get("document_order") if isinstance(nav.get("document_order"), dict) else {}
    page_start, page_end = parse_page_range(metadata.get("pages"))
    original_date = document_date.get("original_text") or metadata.get("date_text")
    normalized_date = document_date.get("normalized_date") or parse_date_text(original_date)
    return {
        "brief_title": nav.get("brief_title") or source.get("title"),
        "navigation_summary": nav.get("navigation_summary") or "",
        "document_date": {
            "original_text": original_date,
            "normalized_date": normalized_date,
            "precision": document_date.get("precision") or ("day" if normalized_date else "unknown"),
        },
        "document_order": {
            "volume": document_order.get("volume") or metadata.get("volume"),
            "page_start": document_order.get("page_start") if document_order.get("page_start") is not None else page_start,
            "page_end": document_order.get("page_end") if document_order.get("page_end") is not None else page_end,
            "sequence_label": document_order.get("sequence_label") or metadata.get("pages"),
        },
        "document_role": nav.get("document_role") or infer_document_role(source),
        "topics": nav.get("topics") or [],
        "primary_people": nav.get("primary_people") or fallback_people(metadata),
        "primary_locations": nav.get("primary_locations") or [],
        "referenced_events": nav.get("referenced_events") or [],
        "claim_count": nav.get("claim_count") or 0,
        "event_reference_count": nav.get("event_reference_count") or len(nav.get("referenced_events") or []),
        "confidence": nav.get("confidence"),
    }


def fallback_people(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    witness = metadata.get("witness_name")
    if not witness:
        return []
    return [
        {
            "label": witness,
            "canonical_id": None,
            "role_or_relationship": "witness",
            "confidence": None,
        }
    ]


def infer_document_role(source: dict[str, Any]) -> str:
    source_type = source.get("source_type")
    if source_type in {"testimony", "exhibit", "toc"}:
        return source_type
    title = str(source.get("title") or "").lower()
    if "cover page" in title:
        return "cover"
    if any(term in title for term in ("summons", "motion", "order", "indictment")):
        return "legal_document"
    return "other"


def parse_page_range(value: Any) -> tuple[int | None, int | None]:
    numbers = [int(match) for match in re.findall(r"\d+", str(value or ""))]
    if not numbers:
        return None, None
    return numbers[0], numbers[-1]


def parse_date_text(value: Any) -> str | None:
    text = str(value or "").strip().replace("Sept.", "Sep.").replace("August", "Aug.")
    if not text:
        return None
    patterns = [
        ("%Y %B %d", r"(\d{4})\s+([A-Za-z]+)\.?\s+(\d{1,2})"),
        ("%Y %b %d", r"(\d{4})\s+([A-Za-z]+)\.?\s+(\d{1,2})"),
        ("%B %d, %Y", r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})"),
        ("%b %d, %Y", r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})"),
    ]
    from datetime import datetime

    for fmt, pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = " ".join(match.groups()) if fmt.startswith("%Y") else f"{match.group(1)} {match.group(2)}, {match.group(3)}"
        try:
            return datetime.strptime(candidate.replace(".", ""), fmt.replace(".", "")).date().isoformat()
        except ValueError:
            continue
    return None


def read_transcript_json(source_id: str) -> dict[str, Any] | None:
    page = _find_transcript_page(source_id)
    if not page or not page.get("transcript_json_path"):
        return None
    return read_json(page["transcript_json_path"])


def read_transcript_tei(source_id: str) -> str | None:
    page = _find_transcript_page(source_id)
    if not page or not page.get("tei_path"):
        return None
    return read_text(page["tei_path"])


def read_transcript_html(source_id: str) -> str | None:
    page = _find_transcript_page(source_id)
    if not page or not page.get("raw_html_path"):
        return None
    return read_text(page["raw_html_path"])


def _find_transcript_page(source_id: str) -> dict[str, Any] | None:
    if not source_id.startswith("source_"):
        return None
    latest_run = read_latest_hadc_run()
    if not latest_run:
        return None
    pages = read_json(f"raw/haymarket/hadc/{latest_run['run_id']}/pages.json") or []
    return next((page for page in pages if page.get("id") == source_id), None)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
