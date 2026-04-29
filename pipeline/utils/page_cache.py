"""Page-level cache for extracted bundles.

Each page's final Stage C bundle (the input to harmonize_bundles) is
keyed by a hash of the pipeline configuration AND the page's source-text
sha256. Same config + same source = cache hit; bumping a prompt
template version, a model name, or a chunking parameter invalidates the
cache automatically.

Cache layout:
    cache/haymarket/<cache_key>/<page_id>/bundle.json
    cache/haymarket/<cache_key>/<page_id>/metadata.json

The cache is durable across runs and intentionally NOT cleaned by
--clear-data (which only wipes per-run audit artifacts). Use
--clear-cache or `rm -rf data/cache` to invalidate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from utils.s3_storage import JsonStorage


CACHE_PREFIX = "cache/haymarket"


def compute_cache_key(
    page: dict[str, Any],
    *,
    briefing_model: str,
    transcription_model: str,
    transcription_format: str,
    tagging_model: str,
    briefing_prompt_template: str,
    transcription_prompt_template: str,
    tagging_prompt_template: str,
) -> str:
    """SHA256 of the pipeline config + source text hash.

    Anything that meaningfully changes the bundle should be in the
    fingerprint. Bumping a prompt template version (e.g.
    'haymarket_segment_tagging_v1' -> '_v2') is the canonical way to
    invalidate cached bundles after a behavior change.
    """
    parts = [
        page.get("candidate_text_sha256") or page.get("id", ""),
        briefing_model,
        transcription_model,
        transcription_format,
        tagging_model,
        briefing_prompt_template,
        transcription_prompt_template,
        tagging_prompt_template,
    ]
    fingerprint = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(fingerprint).hexdigest()[:16]


def cache_paths(cache_key: str, page_id: str) -> dict[str, str]:
    base = f"{CACHE_PREFIX}/{cache_key}/{page_id}"
    return {
        "bundle": f"{base}/bundle.json",
        "metadata": f"{base}/metadata.json",
    }


def read_cached_bundle(
    storage: JsonStorage,
    cache_key: str,
    page_id: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return (bundle, metadata) if both files exist, else None.

    A partial cache entry (bundle without metadata or vice versa) is
    treated as a miss — safer than serving incomplete state.
    """
    paths = cache_paths(cache_key, page_id)
    if not storage.exists(paths["bundle"]):
        return None
    if not storage.exists(paths["metadata"]):
        return None
    bundle = storage.read_json(paths["bundle"])
    metadata = storage.read_json(paths["metadata"])
    return bundle, metadata


def write_cached_bundle(
    storage: JsonStorage,
    cache_key: str,
    page_id: str,
    bundle: dict[str, Any],
    *,
    run_id: str,
    config: dict[str, Any],
    cost_usd: float,
    usage: dict[str, Any],
) -> dict[str, str]:
    """Persist a successful page bundle plus producer metadata.

    Metadata records what the cache cost when first produced (so cache
    hits can show "saved $X" in summaries) and which run wrote it (for
    audit chaining).
    """
    paths = cache_paths(cache_key, page_id)
    metadata = {
        "cache_key": cache_key,
        "page_id": page_id,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by_run_id": run_id,
        "config": config,
        "cost_usd": round(float(cost_usd), 8),
        "usage": dict(usage),
    }
    storage.write_json(paths["bundle"], bundle)
    storage.write_json(paths["metadata"], metadata)
    return paths


def clear_cache(storage: JsonStorage) -> int:
    """Wipe the entire haymarket cache. Returns file count cleared."""
    return storage.clear_prefix(CACHE_PREFIX)


def cache_inventory(data_dir) -> dict[str, Any]:
    """Walk the local cache directory and return per-config summaries.

    Local-disk only for now (LocalJsonStorage). S3 support would need a
    list_prefix abstraction on JsonStorage, but cache-status is a
    debugging tool — we'll add S3 if/when someone needs it.
    """
    from pathlib import Path

    base = Path(data_dir) / CACHE_PREFIX
    if not base.exists():
        return {"base_path": str(base), "configs": [], "total_pages": 0, "total_cost_usd": 0.0}

    configs: list[dict[str, Any]] = []
    total_cost = 0.0
    total_pages = 0

    for cfg_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        entries: list[dict[str, Any]] = []
        cfg_cost = 0.0
        cfg_config: dict[str, Any] | None = None
        for page_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            metadata_path = page_dir / "metadata.json"
            bundle_path = page_dir / "bundle.json"
            if not metadata_path.exists() or not bundle_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
            except Exception:
                continue
            cost = float(metadata.get("cost_usd") or 0.0)
            cfg_cost += cost
            total_cost += cost
            total_pages += 1
            entries.append(
                {
                    "page_id": page_dir.name,
                    "produced_at": metadata.get("produced_at"),
                    "produced_by_run_id": metadata.get("produced_by_run_id"),
                    "cost_usd": round(cost, 8),
                }
            )
            if cfg_config is None:
                cfg_config = metadata.get("config") or {}
        configs.append(
            {
                "cache_key": cfg_dir.name,
                "config": cfg_config or {},
                "page_count": len(entries),
                "cost_usd": round(cfg_cost, 8),
                "pages": entries,
            }
        )

    return {
        "base_path": str(base),
        "configs": configs,
        "total_pages": total_pages,
        "total_cost_usd": round(total_cost, 8),
    }
