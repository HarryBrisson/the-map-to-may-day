from __future__ import annotations

import json
import logging
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
    return [
        {
            "id": page["id"],
            "url": page["url"],
            "title": page["title"],
            "source_type": page["source_type"],
            "transcript_metadata": page.get("transcript_metadata", {}),
            "source_stats": page.get("source_stats", {}),
            "tei_path": page.get("tei_path"),
            "transcript_json_path": page.get("transcript_json_path"),
        }
        for page in pages
    ]


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
