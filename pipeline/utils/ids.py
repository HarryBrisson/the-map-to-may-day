from __future__ import annotations

import re
import unicodedata


def slugify(value: str, prefix: str | None = None) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    if not slug:
        slug = "unknown"
    return f"{prefix}_{slug}" if prefix else slug


def stable_source_id(url: str) -> str:
    cleaned = url.rstrip("/").split("/")[-1].replace(".htm", "")
    return slugify(cleaned, "source_hadc")
