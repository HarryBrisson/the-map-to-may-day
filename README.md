# The Map to May Day

An MVP for exploring Haymarket trial people, places, events, and source-specific claims on an interactive map.

The repo follows the local architecture guide: a standalone Python pipeline writes JSON artifacts, and a standalone Flask/Zappa app reads those artifacts. For the MVP, JSON is local-first under `data/`; S3 can be enabled later with the same path contract.

## Layout

- `app/` - Flask app, templates, static JavaScript/CSS, local/S3 read helpers, Zappa config.
- `pipeline/` - CLI pipeline for pulling HADC pages, running OpenAI structured extraction, auditing LLM cost/output, and writing enriched JSON.
- `docs/s3-schema.md` - JSON artifact contract between pipeline and app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt -r pipeline/requirements.txt
```

Create `.env` from the sample file, or export env vars directly:

```bash
cp .env.example .env
```

Then fill in:

```bash
OPENAI_API_KEY=...
GOOGLE_MAPS_API_KEY=...
HAYMARKET_DATA_DIR=./data
HAYMARKET_STORAGE_BACKEND=local
```

The pipeline loads both repo-root `.env` and `pipeline/.env`; use `pipeline/.env.example` if you prefer pipeline-only secrets. `OPENAI_API_KEY` is required for `enrich` or `all`. `GOOGLE_MAPS_API_KEY` is required only for `geocode` or `--geocode`. The pipeline does not provide a local extraction fallback.

## Run The Test Corpus

Pull a small representative HADC subset:

```bash
python pipeline/main.py --action pull --corpus test --output local
```

Run extraction with one or more OpenAI models:

```bash
python pipeline/main.py --action enrich --corpus test --output local --llm-model gpt-4o-mini gpt-4o
```

Update only the source/document briefs used by the transcript navigator, without rerunning transcription or tagging:

```bash
python pipeline/main.py --action briefs --corpus full --output local --briefing-model gpt-5-mini --max-brief-workers 32
```

Use `--pages source_hadc_i019_052` to update a small subset first. Brief updates resume by default when rerun with the same `--run-id`: successful per-page `briefing.json` audit files are reused and only missing/failed pages call the model. Across run IDs, briefs also use a durable cache keyed by source text hash, briefing model, prompt template, and schema digest. Pass `--fresh-briefs` to ignore both resume files and the durable brief cache, generate fresh briefs, and leave existing cache entries untouched.

Optionally geocode the unique LLM-extracted locations. This first asks the LLM for a modern Google Maps query/address, then calls the Google Maps Geocoding API once per unique `location.id`:

```bash
export GOOGLE_MAPS_API_KEY="..."
python pipeline/main.py --action geocode --output local --geocoder google --geocode-llm-model gpt-4o-mini
```

You can also geocode immediately after extraction:

```bash
python pipeline/main.py --action all --corpus test --output local --llm-model gpt-4o-mini --geocode
```

Or run both:

```bash
python pipeline/main.py --action all --corpus test --output local --llm-model gpt-4o-mini gpt-4o
```

Important outputs:

- `data/raw/haymarket/hadc/<run_id>/pages.json`
- `data/raw/haymarket/llm/<run_id>/<model>/<call_id>.json`
- `data/raw/haymarket/geolocation/<run_id>/llm/<location_id>_<model>_modern_address.json`
- `data/raw/haymarket/geolocation/<run_id>/google/<location_id>_google_geocode.json`
- `data/enriched/haymarket/{people,locations,claims,events,sources}/latest.json`
- `data/enriched/haymarket/llm_costs/<run_id>.json`
- `data/enriched/haymarket/model_evals/<run_id>.json`
- `data/enriched/haymarket/geolocation/<run_id>.json`

## Run The App

```bash
cd app
python app.py
```

Open `http://localhost:5001`. The app reads `../data` by default, or `HAYMARKET_DATA_DIR` when configured.

## Optional S3 Mode

The storage layer supports `--output s3` for the pipeline and `HAYMARKET_STORAGE_BACKEND=s3` for the app.

Required config:

```bash
export HAYMARKET_S3_BUCKET="your-bucket"
export HAYMARKET_S3_PREFIX="optional/prefix"
```

## Tests

```bash
python -m pytest app/tests pipeline/tests
```
