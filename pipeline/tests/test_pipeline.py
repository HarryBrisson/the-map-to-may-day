import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        sys.modules.pop(module_name)

from enrichment import geolocate_locations  # noqa: E402
from enrichment import stage_briefing  # noqa: E402
from enrichment import stage_tagging  # noqa: E402
from enrichment import stage_transcription  # noqa: E402
from enrichment.geolocate_locations import geolocate_locations as geolocate_location_records  # noqa: E402
from enrichment.harmonization import harmonize_bundles  # noqa: E402
from enrichment.pipeline import merge_events, run_enrichment  # noqa: E402
from enrichment.stage_tagging import split_tei_into_units  # noqa: E402
from enrichment.tei_validation import validate_generated_tei  # noqa: E402
from main import clear_generated_data  # noqa: E402
from sources.hadc_source import (  # noqa: E402
    add_standoff_annotations_to_tei,
    build_tei_transcript,
    extract_candidate_core_text,
    extract_content_blocks,
    extract_page_markers,
    tei_to_transcript_json,
    transcript_artifact_paths,
)
from utils import openai_schema  # noqa: E402
from utils.s3_storage import LocalJsonStorage  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402


def test_segment_tags_schema_matches_strict_response_format_subset() -> None:
    schema = stage_tagging.load_segment_tags_schema()
    location_schema = schema["properties"]["locations"]["items"]
    coordinates_schema = location_schema["properties"]["coordinates"]

    assert "geocoding" not in location_schema["properties"]
    assert "provider" not in coordinates_schema["properties"]
    assert "place_id" not in coordinates_schema["properties"]
    assert "raw_response_id" not in coordinates_schema["properties"]
    assert_openai_strict_schema(schema)


def test_briefing_schema_matches_strict_response_format_subset() -> None:
    schema = stage_briefing.load_briefing_schema()
    assert_openai_strict_schema(schema)


def test_extract_tei_from_text_strips_preamble_and_code_fences() -> None:
    fenced = """Sure, here is the TEI:
```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>Hi.</p></body></text></TEI>
```
"""
    extracted = stage_transcription.extract_tei_from_text(fenced)
    assert extracted.startswith("<TEI")
    assert extracted.endswith("</TEI>")
    assert "```" not in extracted


def test_parse_tei_xml_escapes_stray_ampersands() -> None:
    from sources.hadc_source import parse_tei_xml

    # &c. (et cetera) appears verbatim in 19th-century trial transcripts;
    # it would otherwise crash ET.fromstring with "not well-formed (invalid token)"
    tei = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>back to work, &c., and their ends would be defeated</p></body></text></TEI>'
    root = parse_tei_xml(tei)
    p_text = root.find(".//{http://www.tei-c.org/ns/1.0}p").text
    assert "&c." in p_text


def test_extract_tei_from_text_handles_truncated_output() -> None:
    truncated = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>Half a docu'
    extracted = stage_transcription.extract_tei_from_text(truncated)
    assert extracted.startswith("<TEI")


def test_parse_jsonl_rows_skips_garbage_and_keeps_valid_objects() -> None:
    raw = """
    Sure, here are the rows:
    ```jsonl
    {"page":"17","speaker":"Q","speaker_id":"#mr_grinnell","text":"What is your name?"}
    not a json line
    {"page":"17","speaker":"A","speaker_id":"#bonfield","text":"John Bonfield."}
    {"page":"18","speaker":null,"speaker_id":null,"text":"Whereupon a recess..."}
    ```
    """
    rows = stage_transcription.parse_jsonl_rows(raw)
    assert len(rows) == 3
    assert rows[0]["text"] == "What is your name?"
    assert rows[2]["speaker"] is None


def test_tag_one_unit_with_retry_recovers_from_first_failure(monkeypatch) -> None:
    from enrichment.stage_tagging import TaggingUnit, tag_one_unit_with_retry

    calls = {"count": 0}

    def fake_call_openai_structured(model, input_messages, schema, schema_name, max_output_tokens=None):
        del model, input_messages, schema, schema_name, max_output_tokens
        calls["count"] += 1
        if calls["count"] == 1:
            raise stage_tagging.LLMCallError(
                "transient",
                raw_output={"error": "rate limit"},
                usage={"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
            )
        return (
            {"unit_id": "sp_001", "people": [], "locations": [], "claims": [], "events": [], "quotes": []},
            {},
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

    monkeypatch.setattr(stage_tagging, "call_openai_structured", fake_call_openai_structured)

    unit = TaggingUnit(
        unit_id="sp_001",
        kind="sp",
        speaker_id="bonfield",
        page_ref="17",
        text="John Bonfield.",
        tei_snippet="<sp/>",
    )
    result = tag_one_unit_with_retry(
        page={"id": "test_page"},
        briefing=None,
        unit=unit,
        prior_units=[],
        model="gpt-4.1-mini",
        schema={},
        max_attempts=3,
    )

    assert result["status"] == "success"
    assert result["attempts"] == 2
    # Usage accumulates across both attempts (100+100 input)
    assert result["usage"]["input_tokens"] == 200
    assert "prior_errors" in result and len(result["prior_errors"]) == 1


def test_synthesize_tei_from_rows_produces_well_formed_tei() -> None:
    page = {
        "id": "source_hadc_i019_052",
        "page_cues": [{"page_ref": "17", "facs": "https://example/page17.htm"}],
    }
    rows = [
        {"page": "17", "speaker": "Q", "speaker_id": "#mr_grinnell", "text": "What is your name?"},
        {"page": "17", "speaker": "A", "speaker_id": "#bonfield", "text": "John Bonfield."},
        {"page": "18", "speaker": None, "speaker_id": None, "text": "Whereupon a recess was taken."},
    ]
    tei = stage_transcription.synthesize_tei_from_rows(page, rows)
    # Round-trip parses cleanly and contains the expected structure
    from sources.hadc_source import parse_tei_xml, tei_tag

    root = parse_tei_xml(tei)
    sps = root.findall(f".//{tei_tag('sp')}")
    assert len(sps) == 2
    assert sps[0].attrib.get("who") == "#mr_grinnell"
    assert sps[1].attrib.get("who") == "#bonfield"
    pbs = root.findall(f".//{tei_tag('pb')}")
    assert [pb.attrib.get("n") for pb in pbs] == ["17", "18"]
    assert pbs[0].attrib.get("facs") == "https://example/page17.htm"


def test_clear_generated_data_removes_haymarket_artifacts(tmp_path) -> None:
    storage = LocalJsonStorage(tmp_path)
    storage.write_json("raw/haymarket/example.json", {"ok": True})
    storage.write_json("enriched/haymarket/example.json", {"ok": True})
    storage.write_json("raw/other/example.json", {"ok": True})

    cleared = clear_generated_data(storage)

    assert cleared == {"raw/haymarket": 1, "enriched/haymarket": 1}
    assert not storage.exists("raw/haymarket/example.json")
    assert not storage.exists("enriched/haymarket/example.json")
    assert storage.exists("raw/other/example.json")


def test_candidate_core_text_removes_navigation_and_scripts() -> None:
    soup = BeautifulSoup(
        """
        <html><body>
          <p>Direct examination by Mr. Grinnell.</p>
          <a href="next.htm">Go to Next Witness</a> |
          <a href="toc.htm">Return to Trial TOC</a>
          <a href="019I019.htm">[Image, Volume I, Page 19]</a>
          <p>A John Bonfield.</p>
          <script>ignored()</script>
        </body></html>
        """,
        "html.parser",
    )

    text, stats = extract_candidate_core_text(soup)

    assert "Direct examination by Mr. Grinnell." in text
    assert "[Image, Volume I, Page 19]" in text
    assert "A John Bonfield." in text
    assert "Go to Next Witness" not in text
    assert "Return to Trial TOC" not in text
    assert "ignored" not in text
    assert stats["navigation_removed"] >= 2


def assert_openai_strict_schema(schema) -> None:
    unsupported_keys = openai_schema.OPENAI_UNSUPPORTED_SCHEMA_KEYS
    if isinstance(schema, dict):
        assert not unsupported_keys.intersection(schema)

        properties = schema.get("properties")
        if isinstance(properties, dict):
            assert schema.get("additionalProperties") is False
            assert set(schema.get("required", [])) == set(properties)
            for property_schema in properties.values():
                assert_openai_strict_schema(property_schema)

        if "items" in schema:
            assert_openai_strict_schema(schema["items"])

        for key in ["anyOf", "oneOf", "allOf"]:
            for value in schema.get(key, []):
                assert_openai_strict_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            assert_openai_strict_schema(value)


def test_run_enrichment_writes_app_ready_outputs(tmp_path, monkeypatch) -> None:
    storage = LocalJsonStorage(tmp_path)
    run_id = "test_run"
    source_id = "source_hadc_i019_052"
    fetched_at = "2026-04-27T00:00:00+00:00"
    text = "John Bonfield testified about Desplaines Street Station."
    paths = transcript_artifact_paths(run_id, source_id)
    tei_xml = build_tei_transcript(
        source_id=source_id,
        url="https://example.test/I019-052.htm",
        title="Testimony of John Bonfield",
        source_type="testimony",
        fetched_at=fetched_at,
        raw_html_sha256="abc123",
        text=text,
        page_images=[],
    )
    storage.write_text(paths["raw_html"], "<html>fixture</html>")
    storage.write_text(paths["candidate_text"], text)
    storage.write_text(paths["tei"], tei_xml)
    storage.write_json(
        paths["transcript_json"],
        tei_to_transcript_json(
            source_id=source_id,
            url="https://example.test/I019-052.htm",
            title="Testimony of John Bonfield",
            source_type="testimony",
            fetched_at=fetched_at,
            tei_xml=tei_xml,
            transcript_metadata={"volume": "I", "pages": "19-52"},
        ),
    )
    storage.write_json(
        f"raw/haymarket/hadc/{run_id}/pages.json",
        [
            {
                "id": source_id,
                "url": "https://example.test/I019-052.htm",
                "title": "Testimony of John Bonfield",
                "source_type": "testimony",
                "fetched_at": fetched_at,
                "raw_html_sha256": "abc123",
                "candidate_text_sha256": "candidate123",
                "raw_html_path": paths["raw_html"],
                "candidate_text_path": paths["candidate_text"],
                "tei_path": paths["tei"],
                "transcript_json_path": paths["transcript_json"],
                "text": text,
                "links": [],
                "page_images": [],
                "page_cues": [],
                "transcript_metadata": {"volume": "I", "pages": "19-52"},
                "source_stats": {"characters": len(text), "lines": 1},
            }
        ],
    )

    transcription_tei = """<?xml version='1.0' encoding='utf-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="source_hadc_i019_052">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Testimony of John Bonfield</title></titleStmt>
      <publicationStmt><p>Generated by The Map to May Day pipeline.</p></publicationStmt>
      <sourceDesc><p>Original HADC URL: https://example.test/I019-052.htm</p></sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div type="testimony">
        <p><seg xml:id="m_person_john_bonfield" type="person" corresp="#person_john_bonfield">John Bonfield</seg> testified about <seg xml:id="m_location_desplaines_station" type="location" corresp="#location_desplaines_station">Desplaines Street Station</seg>.</p>
      </div>
    </body>
  </text>
  <standOff>
    <listAnnotation>
      <annotation xml:id="quote_bonfield_station" type="quote" target="#m_person_john_bonfield" resp="#person_john_bonfield" cert="0.9">John Bonfield testified about Desplaines Street Station.</annotation>
    </listAnnotation>
  </standOff>
</TEI>"""
    briefing = {
        "source_id": source_id,
        "summary": "Bonfield testifies about Desplaines Street Station.",
        "witness": {"name": "John Bonfield", "role": "police"},
        "examiners": [],
        "defendants_referenced": [],
        "key_locations": ["Desplaines Street Station"],
        "key_dates": [],
        "topics": ["police movement"],
        "speaker_directory": [
            {"speaker_id": "#bonfield", "display_name": "John Bonfield", "role": "witness"},
        ],
    }
    tagging_payload = {
        "unit_id": "p_000",
        "people": [
            {
                "id": "person_john_bonfield",
                "display_name": "John Bonfield",
                "alternate_names": ["Inspector Bonfield"],
                "roles": ["witness", "police"],
                "bio": {
                    "birth_year": 1836,
                    "death_year": 1898,
                    "occupation": "Inspector of Police",
                    "summary": "Police inspector who testified for the prosecution.",
                },
                "source_ids": [source_id],
                "confidence": 0.95,
            }
        ],
        "locations": [
            {
                "id": "location_desplaines_station",
                "name": "Desplaines Street Station",
                "address_1886": "Desplaines Street Station, Chicago",
                "address_1887": None,
                "modern_address": None,
                "coordinates": {
                    "lat": 41.88295,
                    "lng": -87.64445,
                    "confidence": 0.35,
                    "method": "llm_fixture",
                },
                "location_type": "police_station",
                "source_ids": [source_id],
                "notes": "Police rendezvous point.",
            }
        ],
        "claims": [
            {
                "id": "claim_bonfield_station_1886_05_04_1800",
                "claim_type": "presence_or_movement",
                "source_id": source_id,
                "reported_by_person_id": "person_john_bonfield",
                "claim_made_at": "1886-07-16",
                "subject_person_ids": ["person_john_bonfield"],
                "location_id": "location_desplaines_station",
                "event_time": {
                    "start": "1886-05-04T18:00:00",
                    "end": None,
                    "precision": "approximate",
                    "original_text": "in the vicinity of six o'clock",
                },
                "statement": "Bonfield said he arrived at Desplaines Street Station around six o'clock.",
                "quote": "in the vicinity of six o'clock",
                "page_refs": ["Volume I, page 19"],
                "confidence": 0.88,
            }
        ],
        "events": [
            {
                "id": "event_police_rendezvous_desplaines_station",
                "title": "Police rendezvous at Desplaines Street Station",
                "description": "Police assembled at Desplaines Street Station.",
                "event_type": "assembly",
                "time": {
                    "start": "1886-05-04T18:00:00",
                    "end": None,
                    "precision": "approximate",
                    "display": "Evening of May 4, 1886",
                },
                "location_id": "location_desplaines_station",
                "participant_person_ids": ["person_john_bonfield"],
                "claim_ids": ["claim_bonfield_station_1886_05_04_1800"],
                "disputed_fields": [],
                "confidence": 0.82,
            }
        ],
        "quotes": [
            {
                "id": "quote_bonfield_station",
                "source_id": source_id,
                "speaker_person_id": "person_john_bonfield",
                "speaker_label": "John Bonfield",
                "quote": "John Bonfield testified about Desplaines Street Station.",
                "page_refs": [],
                "confidence": 0.9,
            }
        ],
    }
    usage = {"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}

    def structured_response(model, input_messages, schema, schema_name, max_output_tokens=None):
        del model, input_messages, schema, max_output_tokens
        if schema_name == "haymarket_page_briefing":
            return briefing, {"output": "structured"}, usage
        if schema_name == "haymarket_segment_tags":
            return tagging_payload, {"output": "structured"}, usage
        raise AssertionError(f"unexpected schema_name: {schema_name}")

    def text_response(model, input_messages, max_output_tokens=None):
        del model, input_messages, max_output_tokens
        return transcription_tei, {"output": "text"}, usage

    monkeypatch.setattr(stage_briefing, "call_openai_structured", structured_response)
    monkeypatch.setattr(stage_transcription, "call_openai_text", text_response)
    monkeypatch.setattr(stage_tagging, "call_openai_structured", structured_response)

    result = run_enrichment(
        storage=storage,
        run_id=run_id,
        corpus="test",
        llm_provider="openai",
        llm_models=["gpt-4o-mini", "gpt-4o"],
        streaming=False,
    )

    assert len(result["people"]) == 1
    assert len(result["locations"]) == 1
    assert len(result["claims"]) == 1
    assert len(result["events"]) == 1
    assert storage.exists("enriched/haymarket/llm_costs/test_run.json")
    assert storage.exists("enriched/haymarket/model_evals/test_run.json")
    assert storage.exists("enriched/haymarket/claims/latest.json")
    transcript = storage.read_json(paths["transcript_json"])
    assert any(mention["entity_id"] == "person_john_bonfield" for mention in transcript["mentions"])
    assert any(mention["entity_id"] == "location_desplaines_street_station" for mention in transcript["mentions"])
    assert result["quotes"][0]["speaker_person_id"] == "person_john_bonfield"


def test_claims_are_not_promoted_to_events_without_event_suggestions() -> None:
    claims = [
        {
            "id": "claim_diagram_evidence",
            "claim_type": "evidence",
            "location_id": "location_haymarket_square",
            "event_time": {"start": None, "end": None, "precision": "exact", "original_text": "1886 July 16"},
            "subject_person_ids": [],
            "confidence": 1.0,
            "statement": "Diagram introduced into evidence.",
        }
    ]

    assert merge_events([], claims) == []


def test_transcription_prompt_includes_speaker_context_and_full_text() -> None:
    page = {
        "id": "source_hadc_i019_052",
        "url": "https://example.test/I019-052.htm",
        "title": "Testimony of John Bonfield",
        "source_type": "testimony",
        "text": "[Image, Volume I, Page 19]\nQ What is your name?\nA John Bonfield.",
        "transcript_metadata": {"witness_name": "John Bonfield"},
        "source_stats": {"page_markers": 1},
        "page_cues": [{"index": 0, "line_index": 0, "label": "Volume I, Page 19", "page_ref": "19", "facs": "page19.htm"}],
        "toc_entries": [],
        "tei_path": "tei.xml",
        "transcript_json_path": "transcript.json",
    }
    briefing = {
        "summary": "Bonfield testifies.",
        "witness": {"name": "John Bonfield", "role": "police"},
        "examiners": [{"name": "Mr. Grinnell", "side": "prosecution"}],
        "speaker_directory": [
            {"speaker_id": "#bonfield", "display_name": "John Bonfield", "role": "witness"},
            {"speaker_id": "#grinnell", "display_name": "Mr. Grinnell", "role": "prosecutor"},
        ],
    }

    chunk = stage_transcription.build_source_chunks(page, "<html>fixture</html>")[0]
    messages = stage_transcription.build_transcription_messages(page, briefing, chunk)
    user_content = messages[1]["content"]

    assert "A = John Bonfield" not in user_content
    assert '"answer_speaker": "John Bonfield"' in user_content
    assert "Q What is your name?" in user_content
    assert "#bonfield" in user_content
    assert "speaker_directory" in user_content


def test_split_tei_into_units_uses_sp_when_available() -> None:
    tei_xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body><div>
    <pb n="19"/>
    <sp who="#grinnell"><speaker>Q.</speaker><p>What is your name?</p></sp>
    <sp who="#bonfield"><speaker>A.</speaker><p>John Bonfield.</p></sp>
  </div></body></text>
</TEI>"""
    units = split_tei_into_units(tei_xml)
    assert len(units) == 2
    assert units[0].kind == "sp"
    assert units[0].speaker_id == "grinnell"
    assert units[1].speaker_id == "bonfield"
    assert units[0].page_ref == "19"
    assert "What is your name?" in units[0].text
    assert "John Bonfield" in units[1].text


def test_split_tei_into_units_falls_back_to_paragraphs() -> None:
    tei_xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body><div type="exhibit">
    <pb n="10"/>
    <p>Diagram of the Haymarket area.</p>
    <p>Marked exhibit ten.</p>
  </div></body></text>
</TEI>"""
    units = split_tei_into_units(tei_xml)
    assert len(units) == 2
    assert all(unit.kind == "p" for unit in units)
    assert units[0].page_ref == "10"


def test_token_recall_passes_when_words_match_but_order_differs() -> None:
    page = {
        "id": "source_hadc_i019_052",
        "url": "https://example.test/I019-052.htm",
        "title": "Testimony",
        "source_type": "testimony",
        "fetched_at": "2026-04-27T00:00:00+00:00",
        "text": "John Bonfield testified about Desplaines Street Station.",
        "page_cues": [],
        "transcript_metadata": {},
    }
    reflowed_tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><p>About Desplaines Street Station, John Bonfield testified.</p></div></body></text></TEI>"""

    validation = validate_generated_tei(page, reflowed_tei)
    assert validation["text"]["token_recall"] >= 0.95
    assert validation["status"] == "valid"


def test_generated_tei_validation_detects_text_drift() -> None:
    page = {
        "id": "source_hadc_i019_052",
        "url": "https://example.test/I019-052.htm",
        "title": "Testimony of John Bonfield",
        "source_type": "testimony",
        "fetched_at": "2026-04-27T00:00:00+00:00",
        "text": "John Bonfield testified about Desplaines Street Station.",
        "page_cues": [],
        "transcript_metadata": {},
    }
    valid_tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><p>John Bonfield testified about Desplaines Street Station.</p></div></body></text></TEI>"""
    drifted_tei = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><p>Different text.</p></div></body></text></TEI>"""

    assert validate_generated_tei(page, valid_tei)["status"] == "valid"
    assert validate_generated_tei(page, drifted_tei)["status"] == "invalid"


def test_harmonization_rewrites_provisional_ids_and_quotes() -> None:
    bundle = {
        "source_id": "source_hadc_i019_052",
        "tei_xml": """<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><sp who="#provisional_person_bonfield"><speaker>A</speaker><p><seg xml:id="m1" type="location" corresp="#provisional_location_station">Desplaines Street Station</seg></p></sp></div></body></text></TEI>""",
        "people": [
            {
                "id": "provisional_person_bonfield",
                "display_name": "John Bonfield",
                "alternate_names": [],
                "roles": ["witness"],
                "bio": {"birth_year": None, "death_year": None, "occupation": None, "summary": None},
                "source_ids": ["source_hadc_i019_052"],
                "confidence": 0.9,
            }
        ],
        "locations": [
            {
                "id": "provisional_location_station",
                "name": "Desplaines Street Station",
                "address_1886": "Desplaines Street Station",
                "address_1887": None,
                "modern_address": None,
                "coordinates": {"lat": None, "lng": None, "confidence": 0, "method": "llm"},
                "location_type": "police_station",
                "source_ids": ["source_hadc_i019_052"],
                "notes": None,
            }
        ],
        "claims": [
            {
                "id": "claim_1",
                "claim_type": "presence",
                "source_id": "source_hadc_i019_052",
                "reported_by_person_id": "provisional_person_bonfield",
                "claim_made_at": "1886-07-16",
                "subject_person_ids": ["provisional_person_bonfield"],
                "location_id": "provisional_location_station",
                "event_time": {"start": None, "end": None, "precision": "unknown", "original_text": None},
                "statement": "Bonfield mentioned the station.",
                "quote": "Desplaines Street Station",
                "page_refs": [],
                "confidence": 0.8,
            }
        ],
        "event_suggestions": [],
        "quotes": [
            {
                "id": "quote_1",
                "source_id": "source_hadc_i019_052",
                "speaker_person_id": "provisional_person_bonfield",
                "speaker_label": "John Bonfield",
                "quote": "Desplaines Street Station",
                "page_refs": [],
                "confidence": 0.8,
            }
        ],
    }

    harmonized = harmonize_bundles([bundle])

    assert harmonized["claims"][0]["reported_by_person_id"] == "person_john_bonfield"
    assert harmonized["claims"][0]["location_id"] == "location_desplaines_street_station"
    assert harmonized["quotes"][0]["speaker_person_id"] == "person_john_bonfield"
    assert 'who="#person_john_bonfield"' in harmonized["bundles"][0]["tei_xml"]
    assert 'corresp="#location_desplaines_street_station"' in harmonized["bundles"][0]["tei_xml"]


def test_hadc_source_tei_and_transcript_json_preserve_page_structure() -> None:
    text = "\n".join(
        [
            "HADC - Testimony of John Bonfield",
            "[Image, Volume I, Page 19]",
            "Q. What is your name?",
            "A John Bonfield.",
            "[Image, Volume I, Page 20]",
            "MR. GRINNELL: I will have the circular here.",
        ]
    )

    markers = extract_page_markers(text)
    blocks = extract_content_blocks(text)
    tei_xml = build_tei_transcript(
        source_id="source_hadc_i019_052",
        url="https://example.test/I019-052.htm",
        title="Testimony of John Bonfield",
        source_type="testimony",
        fetched_at="2026-04-27T00:00:00+00:00",
        raw_html_sha256="abc123",
        text=text,
        page_images=["https://example.test/page19.htm", "https://example.test/page20.htm"],
    )
    transcript = tei_to_transcript_json(
        source_id="source_hadc_i019_052",
        url="https://example.test/I019-052.htm",
        title="Testimony of John Bonfield",
        source_type="testimony",
        fetched_at="2026-04-27T00:00:00+00:00",
        tei_xml=tei_xml,
    )

    assert [marker["page_ref"] for marker in markers] == ["19", "20"]
    assert [block["kind"] for block in blocks][2:4] == ["question", "answer"]
    assert [page["page_ref"] for page in transcript["page_refs"]] == ["19", "20"]
    assert [segment["kind"] for segment in transcript["segments"]][1:3] == ["question", "answer"]
    start = transcript["text"].index("John Bonfield")
    annotated = add_standoff_annotations_to_tei(
        tei_xml,
        [
            {
                "id": "mention_person_john_bonfield",
                "kind": "person",
                "entity_id": "person_john_bonfield",
                "start": start,
                "end": start + len("John Bonfield"),
                "text": "John Bonfield",
                "confidence": 0.95,
                "source": "test",
            }
        ],
    )
    annotated_transcript = tei_to_transcript_json(
        source_id="source_hadc_i019_052",
        url="https://example.test/I019-052.htm",
        title="Testimony of John Bonfield",
        source_type="testimony",
        fetched_at="2026-04-27T00:00:00+00:00",
        tei_xml=annotated,
    )
    assert annotated_transcript["mentions"][0]["entity_id"] == "person_john_bonfield"


def test_geolocation_runs_once_per_location_and_stores_raw_artifacts(tmp_path, monkeypatch) -> None:
    storage = LocalJsonStorage(tmp_path)
    locations = [
        {
            "id": "location_haymarket_square",
            "name": "Haymarket Square",
            "address_1886": "between Desplaines and Randolph streets",
            "address_1887": None,
            "modern_address": None,
            "coordinates": {"lat": None, "lng": None, "confidence": 0, "method": "llm_extracted"},
            "location_type": "square",
            "source_ids": ["source_hadc_i019_052"],
            "notes": None,
        },
        {
            "id": "location_haymarket_square",
            "name": "Haymarket Square duplicate",
            "address_1886": "between Desplaines and Randolph streets",
            "address_1887": None,
            "modern_address": None,
            "coordinates": {"lat": None, "lng": None, "confidence": 0, "method": "llm_extracted"},
            "location_type": "square",
            "source_ids": ["source_hadc_x0010"],
            "notes": None,
        },
    ]

    def location_llm(model, input_messages):
        del model, input_messages
        return (
            {
                "location_id": "location_haymarket_square",
                "modern_query": "W Randolph St and N Desplaines St, Chicago, IL",
                "modern_address": "W Randolph St and N Desplaines St, Chicago, IL",
                "reasoning": "Historic Haymarket was described near Desplaines and Randolph.",
                "confidence": 0.7,
            },
            {"output": "structured"},
            {"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
        )

    def google(location, modernized, storage, run_id, api_key):
        del api_key
        record = {
            "run_id": run_id,
            "response_id": f"{location['id']}_google_geocode",
            "timestamp": "2026-04-27T00:00:00+00:00",
            "provider": "google_maps",
            "location_id": location["id"],
            "query": modernized["modern_query"],
            "raw_output": {
                "status": "OK",
                "results": [
                    {
                        "formatted_address": "W Randolph St & N Desplaines St, Chicago, IL 60661, USA",
                        "place_id": "test-place",
                        "geometry": {
                            "location": {"lat": 41.88415, "lng": -87.64435},
                            "location_type": "GEOMETRIC_CENTER",
                        },
                    }
                ],
            },
            "status": "success",
            "error": None,
        }
        storage.write_json(f"raw/haymarket/geolocation/{run_id}/google/{record['response_id']}.json", record)
        return record

    monkeypatch.setattr(geolocate_locations, "call_openai_location_modernization", location_llm)
    monkeypatch.setattr(geolocate_locations, "geocode_with_google", google)

    updated, summary = geolocate_location_records(
        locations=locations,
        storage=storage,
        run_id="geocode_test",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        geocoder="google",
        google_api_key="test-key",
    )

    assert len(updated) == 1
    assert summary["locations"] == 1
    assert summary["successes"] == 1
    assert updated[0]["coordinates"]["lat"] == 41.88415
    assert updated[0]["coordinates"]["method"] == "google_maps_geocoding"
    assert storage.exists("raw/haymarket/geolocation/geocode_test/llm/location_haymarket_square_gpt_4o_mini_modern_address.json")
    assert storage.exists("raw/haymarket/geolocation/geocode_test/google/location_haymarket_square_google_geocode.json")
