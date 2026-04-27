import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))
for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        sys.modules.pop(module_name)

from app import app  # noqa: E402


def test_health() -> None:
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_map_data_reads_local_json(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    people_path = data_root / "enriched/haymarket/people/latest.json"
    events_path = data_root / "enriched/haymarket/events/latest.json"
    claims_path = data_root / "enriched/haymarket/claims/latest.json"
    locations_path = data_root / "enriched/haymarket/locations/latest.json"
    sources_path = data_root / "enriched/haymarket/sources/latest.json"
    for path in [people_path, events_path, claims_path, locations_path, sources_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setenv("HAYMARKET_STORAGE_BACKEND", "local\\")
    monkeypatch.setenv("HAYMARKET_DATA_DIR", str(data_root))

    client = app.test_client()
    response = client.get("/api/map-data")

    assert response.status_code == 200
    assert response.get_json() == {"people": [], "events": [], "claims": [], "locations": [], "sources": []}


def test_transcript_api_reads_local_tei_and_json(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    run_id = "test_run"
    source_id = "source_hadc_i019_052"
    transcript_path = f"raw/haymarket/hadc/{run_id}/transcripts/{source_id}.json"
    tei_path = f"raw/haymarket/hadc/{run_id}/tei/{source_id}.xml"
    html_path = f"raw/haymarket/hadc/{run_id}/html/{source_id}.html"

    def write_json(relative_path: str, value) -> None:
        path = data_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_text(relative_path: str, value: str) -> None:
        path = data_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    for dataset in ["people", "events", "claims", "locations", "sources"]:
        write_json(f"enriched/haymarket/{dataset}/latest.json", [])
    write_json("raw/haymarket/hadc/latest_run.json", {"run_id": run_id, "corpus": "test"})
    write_json(
        f"raw/haymarket/hadc/{run_id}/pages.json",
        [
            {
                "id": source_id,
                "url": "https://example.test/source.htm",
                "title": "Testimony of John Bonfield",
                "source_type": "testimony",
                "transcript_metadata": {"volume": "I"},
                "source_stats": {"lines": 1},
                "raw_html_path": html_path,
                "tei_path": tei_path,
                "transcript_json_path": transcript_path,
            }
        ],
    )
    write_json(
        transcript_path,
        {
            "source_id": source_id,
            "title": "Testimony of John Bonfield",
            "source_type": "testimony",
            "text": "John Bonfield.",
            "segments": [{"index": 0, "kind": "text", "page_ref": None, "line": 0, "start": 0, "end": 14, "text": "John Bonfield."}],
            "mentions": [],
        },
    )
    write_text(tei_path, "<TEI><text><body><p>John Bonfield.</p></body></text></TEI>")
    write_text(html_path, "<html>John Bonfield.</html>")

    monkeypatch.setenv("HAYMARKET_STORAGE_BACKEND", "local")
    monkeypatch.setenv("HAYMARKET_DATA_DIR", str(data_root))

    client = app.test_client()
    catalog_response = client.get("/api/transcripts")
    transcript_response = client.get(f"/api/transcripts/{source_id}")
    tei_response = client.get(f"/api/transcripts/{source_id}/tei")
    page_response = client.get("/transcripts")

    assert page_response.status_code == 200
    assert catalog_response.status_code == 200
    assert catalog_response.get_json()[0]["id"] == source_id
    assert transcript_response.status_code == 200
    assert transcript_response.get_json()["source_id"] == source_id
    assert tei_response.status_code == 200
    assert "John Bonfield" in tei_response.get_data(as_text=True)
