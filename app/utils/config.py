from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in Lambda/runtime smoke checks
    load_dotenv = None


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent

if load_dotenv:
    load_dotenv(APP_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env")

_CREDS_CACHE: dict[str, Any] | None = None


def _candidate_creds_paths() -> list[Path]:
    return [
        APP_ROOT / "creds.json",
        REPO_ROOT / "creds.json",
        Path("/var/task/creds.json"),
        Path.cwd() / "creds.json",
    ]


def _load_creds() -> dict[str, Any]:
    global _CREDS_CACHE
    if _CREDS_CACHE is not None:
        return _CREDS_CACHE

    for path in _candidate_creds_paths():
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                _CREDS_CACHE = json.load(handle)
                return _CREDS_CACHE

    _CREDS_CACHE = {}
    return _CREDS_CACHE


def get_config(key: str, default: Any = None, required: bool = False) -> Any:
    value = os.getenv(key)
    if value not in (None, ""):
        return value

    creds = _load_creds()
    if key in creds and creds[key] not in (None, ""):
        return creds[key]

    if required:
        raise RuntimeError(f"Missing required config value: {key}")
    return default


def get_bool_config(key: str, default: bool = False) -> bool:
    value = get_config(key, None)
    if value is None:
        return default
    return _clean_config_string(value).lower() in {"1", "true", "yes", "on"}


def get_flask_secret_key() -> str:
    return str(get_config("FLASK_SECRET_KEY", secrets.token_urlsafe(32)))


def get_storage_backend() -> str:
    return _clean_config_string(get_config("HAYMARKET_STORAGE_BACKEND", "local")).lower().rstrip("\\")


def get_data_root() -> Path:
    configured = get_config("HAYMARKET_DATA_DIR", None)
    if configured:
        return Path(_clean_config_string(configured).rstrip("\\")).expanduser().resolve()
    return REPO_ROOT / "data"


def get_s3_bucket() -> str:
    return _clean_config_string(get_config("HAYMARKET_S3_BUCKET", "", required=get_storage_backend() == "s3"))


def get_s3_prefix() -> str:
    return _clean_config_string(get_config("HAYMARKET_S3_PREFIX", "")).strip("/")


def _clean_config_string(value: Any) -> str:
    return str(value).strip().strip('"').strip("'")
