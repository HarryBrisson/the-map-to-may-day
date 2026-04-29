# JSON Storage Contract

The MVP writes locally under `data/`. The same relative paths are used when the storage backend is S3.

## Raw HADC Pages

Path: `raw/haymarket/hadc/<run_id>/pages.json`

Array of compact source-page records:

- `id`
- `url`
- `title`
- `source_type`
- `fetched_at`
- `raw_html_sha256`
- `raw_html_path`
- `tei_path`
- `transcript_json_path`
- `text`
- `links`
- `page_images`
- `toc_entries`
- `transcript_metadata`
- `source_stats`

The bulky source text derivatives live in sidecar artifacts rather than duplicated arrays inside `pages.json`:

- Original HTML: `raw/haymarket/hadc/<run_id>/html/<source_id>.html`
- TEI transcript: `raw/haymarket/hadc/<run_id>/tei/<source_id>.xml`
- Derived transcript JSON: `raw/haymarket/hadc/<run_id>/transcripts/<source_id>.json`

The TEI transcript is the canonical enhanced text artifact. The derived transcript JSON is generated from TEI for app rendering, search, LLM prompts, and validation. It includes:

- `source_id`
- `text`
- `page_refs`
- `segments`
- `mentions`
- `transcript_metadata`
- `source_stats`

Latest run pointer: `raw/haymarket/hadc/latest_run.json`.

## Raw LLM Calls

Path: `raw/haymarket/llm/<run_id>/<model>/<call_id>.json`

Each file records one LLM call:

- `run_id`
- `call_id`
- `timestamp`
- `provider`
- `model`
- `prompt_template`
- `input_messages`
- `raw_output`
- `parsed_output`
- `source_urls`
- `usage`
- `cost_usd`
- `status`
- `error`

Raw output is stored even when parsing fails when the provider returns a response. Call failures still write an audit file with `status: "error"`.

## Enriched App Datasets

The Flask app reads these files:

- `enriched/haymarket/people/latest.json`
- `enriched/haymarket/locations/latest.json`
- `enriched/haymarket/claims/latest.json`
- `enriched/haymarket/events/latest.json`
- `enriched/haymarket/sources/latest.json`
- `enriched/haymarket/manifest/latest.json`

### Person

People are durable entities with roles and bio details. See `pipeline/schemas/person.schema.json`.

### Location

Locations store historic addresses, modern address if known, and coordinate confidence. See `pipeline/schemas/location.schema.json`.

### Claim

Claims are source-specific assertions. They preserve who reported the claim, when the claim was made, who or what the claim concerns, time, place, quote, and confidence. See `pipeline/schemas/claim.schema.json`.

The app-ready claim records also include a `source` object added during enrichment for display:

- `source.id`
- `source.title`
- `source.url`
- `source.metadata`

### Event

Events are normalized historical happenings assembled from one or more claims. They retain `claim_ids` so disputed testimony can be inspected. See `pipeline/schemas/event.schema.json`.

### Source

Sources preserve the HADC page metadata used by the transcript navigator. Each source summary includes the old path fields plus a `navigation` object derived from the Stage A brief when available, with metadata fallbacks when it is not:

- `navigation.document_date`
- `navigation.document_order`
- `navigation.document_role`
- `navigation.brief_title`
- `navigation.navigation_summary`
- `navigation.topics`
- `navigation.primary_people`
- `navigation.primary_locations`
- `navigation.referenced_events`
- `navigation.claim_count`
- `navigation.event_reference_count`
- `navigation.confidence`

`navigation.referenced_events` are document-level references for browsing and evidence support. They do not replace `enriched/haymarket/events/latest.json`, which remains the canonical normalized event dataset.

## LLM Cost And Model Evaluation

Cost summary path: `enriched/haymarket/llm_costs/<run_id>.json`

Model comparison path: `enriched/haymarket/model_evals/<run_id>.json`

Model evaluations include parse success rate, schema error count, extraction counts, missing required fields, and total estimated cost per model.

## Geolocation

Geolocation is an optional enrichment stage that runs once per unique `location.id`.

The stage has two provider-backed steps:

- LLM modernization: convert historical location fields into a modern Google Maps query/address.
- Google Maps geocoding: call the Google Maps Geocoding API with that query and merge the returned latitude/longitude into the location.

Raw LLM path:

`raw/haymarket/geolocation/<run_id>/llm/<location_id>_<model>_modern_address.json`

Raw Google path:

`raw/haymarket/geolocation/<run_id>/google/<location_id>_google_geocode.json`

Summary path:

`enriched/haymarket/geolocation/<run_id>.json`

The geocoding stage may add these app-facing fields to a location:

- `modern_address`
- `coordinates.provider`
- `coordinates.place_id`
- `coordinates.raw_response_id`
- `geocoding.status`
- `geocoding.query`
- `geocoding.formatted_address`
- `geocoding.llm_call_id`
- `geocoding.google_response_id`
