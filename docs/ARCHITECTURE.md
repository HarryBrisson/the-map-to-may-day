# Architecture Guidelines

A reference for how apps and data pipelines are structured in this ecosystem. Follow these so new work feels familiar.

The repo is organized as **a set of independent CLI pipelines that write to S3**, plus **a Zappa-deployed Flask app that reads from S3**. The two halves share no Python code — they communicate exclusively through JSON files at known S3 paths.

---

## Part 1 — Zappa App Conventions

### Deployment (`zappa_settings.json`)
- Single `prod` stage (no `dev`/`stage` clutter — local dev runs Flask directly).
- AWS region: `us-east-1`.
- App entrypoint: `app.app` (Flask app object lives in `app.py` at the project root).
- `keep_warm: true` with `rate(5 minutes)` — cold starts are unacceptable for the dashboards.
- Custom domain + ACM cert configured directly in `zappa_settings.json` (no separate Terraform/CDK).
- `exclude` list trims `boto3`, `botocore`, `dateutil`, `s3transfer`, `concurrent` (provided by Lambda runtime).
- S3 deployment bucket follows a `<project>-prod` / `holistic-*` naming pattern.

### Project layout
```
<app_name>/
├── app.py                # Flask app + routes, single file
├── zappa_settings.json
├── requirements.txt
├── creds.json.example    # template; real creds.json is gitignored
├── utils/                # all helpers live here
│   ├── config.py         # config loader (see below)
│   ├── auth.py           # Google OAuth + @require_auth decorator
│   ├── s3_utils.py       # all S3 reads/writes
│   └── ...               # one file per data domain
├── static/
│   ├── css/styles.css
│   └── js/
│       ├── pages/<page>.js     # one JS file per route/page
│       ├── charts/<chart>.js   # reusable chart modules
│       └── common/{api,utils}.js
├── templates/
│   ├── base.html
│   ├── <page>.html
│   └── <section>/<page>.html   # grouped by domain
└── docs/                 # ARCHITECTURE, CHANGELOG, styleguide
```

### Configuration (`utils/config.py`)
- Three-tier resolution, in order: **env var → `creds.json` → default**.
- `creds.json` is searched in multiple paths (project root, parent dir, `/var/task/` for Lambda, cwd).
- Single `get_config(key, default=None, required=False)` accessor; thin typed wrappers (`get_plaid_secret()`, `get_oura_client_id()`, etc.) per integration.
- `.env` is loaded via `python-dotenv` for local dev.
- Flask secret key auto-generates if not set.

### Auth (`utils/auth.py`)
- Google OAuth 2.0 via `google-auth-oauthlib`.
- Prod is detected via Lambda environment (use `os.getenv('AWS_LAMBDA_FUNCTION_NAME')`).
- Redirect URI flips between `localhost:5001` (dev) and the prod domain accordingly.
- Allow-list of users (single-tenant by default) sourced from config, not hardcoded in `auth.py`.
- Credentials and `user_email` stored in Flask session; scopes validated on every request to force re-auth on scope change.
- Routes protected with `@require_auth` decorator → redirects to `/login` or `/access-denied`.

### Local dev
- Port **5001** (avoids macOS AirPlay on 5000).
- Run `python app.py` directly; `debug=True` only when invoked as `__main__`.
- Same code path as Lambda; environment differences come from config resolution, not code branches.

### Data
- All persistent data lives in S3 (no RDS/DynamoDB for app data).
- User-scoped paths: `s3://<bucket>/user_data/<user_email>/...`.
- Presigned URLs (1-hour expiry) for any user-facing assets like images/logos.

### Stack
- **Flask 3+**, **Zappa**, **boto3**, **python-dotenv**, **google-auth(-oauthlib)**, **requests**.
- Add integration libs (e.g. `plaid-python`) only when used.
- Frontend: vanilla JS + Plotly, no build step, no framework.

### Documentation expectations
Each app ships with `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`, and `docs/styleguide.md`. README covers setup, env vars, running locally, and an architecture sketch.

### App-level recommendations (small, non-alienating)
- **Bump Python past 3.9.** It's EOL'd; 3.11 or 3.12 on Lambda gets faster cold starts and is a one-line change in `zappa_settings.json`.
- **Pin or upper-bound requirements.** `Flask>=3.0,<4` instead of `Flask>=3.0.0` avoids surprise breakage on a future redeploy.
- **Use `os.getenv('AWS_LAMBDA_FUNCTION_NAME')`** for prod detection instead of inspecting the file's directory path.
- **Move the auth allow-list to `creds.json`/env**, accessed via `get_config()`. Adding a user becomes a config change, not a deploy.
- **Add a `/health` endpoint and point `keep_warm_path` at it.** Avoids the heavy default route running every 5 minutes.
- **Swap `print()` for `logging`** with module-level `logger = logging.getLogger(__name__)`. Same CloudWatch output, but with levels and filterability.
- **Promote inline `render_template_string('<h1>...</h1>')` error pages to real templates** registered with `@app.errorhandler`, so they inherit `base.html` styling.
- **Add a `tests/` folder** with at least a smoke test (`client.get('/health')`) so it's obvious where tests go.
- **Optional: `uv` or `pip-tools` lockfile** (`requirements.lock`) for reproducible deploys. Skip if it feels like ceremony.

---

## Part 2 — Pipeline Conventions

Each data domain (`accounting/`, `biometrics/`, `journaling/`, `media/`, …) is a self-contained CLI pipeline that writes to S3.

### Folder shape (consistent across all domains)
```
<domain>/
├── main.py              # CLI orchestrator (argparse)
├── sources/             # one file per external provider
│   ├── <provider>_source.py
│   └── ...
├── enrichment/          # transform / merge / score
│   ├── pipeline.py      # orchestration
│   └── <step>.py        # individual analyzers (sentiment, scoring, etc.)
├── summary/             # optional: final aggregations (e.g., accounting/summary)
├── utils/
│   ├── s3_storage.py    # S3 read/write helpers
│   └── ...
└── __init__.py
```

### Two-stage flow: **pull → enrich**
- **Pull**: `sources/*` hit external APIs (Withings, Oura, Plaid, Google Drive, Spotify, TMDB, Perplexity, …) and write raw JSON to S3.
- **Enrich**: `enrichment/pipeline.py` reads raw, merges across providers, computes aggregates / derived fields, writes enriched JSON to S3.
- `main.py` exposes both via `--action {pull, enrich, all}`.

### S3 key convention
```
s3://<bucket>/user_data/<user_email>/<stage>/<domain>/<provider>/<year>/<MM>.json
                                     ^^^^^^                                   ^^
                                     raw|enriched                             monthly partition
```
- Data is **partitioned monthly** (`MM.json`).
- `summary/` outputs go to a separate path (`summaries/<year>/...`) — that's typically what the Zappa app reads.
- `get_user_path(user_email, stage, domain, provider, year, file)` is the standard key builder.

### CLI conventions (`main.py`)
Every domain's `main.py` accepts the same flag set, so operating them feels identical:
- `--user <email>` (required)
- `--year` / `--month` (explicit)
- `--current-month` / `--previous-month` (convenience for cron)
- `--start-date YYYY-MM` / `--end-date YYYY-MM` (historical backfill — loops month-by-month)
- `--action {pull, enrich, all}` (default: `all`)

Standard banner output with `=` rules, emoji status indicators (`✅ ❌ 📊 📈 📅`), and a `try/except` around the whole run that prints a traceback and `sys.exit(1)` on failure.

### Imports
- Each domain is run as a script, not installed. `main.py` and `enrichment/pipeline.py` start with `sys.path.insert(0, os.path.dirname(...))` so sibling subpackages resolve.
- No cross-domain imports — `accounting` doesn't import from `biometrics`. Domains communicate exclusively through S3.

### App ↔ pipeline boundary
- **Pipelines write, app reads.** That is the entire contract.
- The app's `utils/s3_utils.py` is the read-side; each pipeline's `<domain>/utils/s3_storage.py` is the write-side. **Intentional duplication** — keeps deploy boundaries clean (the Zappa Lambda doesn't ship pipeline code, and pipelines don't depend on Flask).
- Shared schema lives in S3 JSON, not in code.

### Inline tests
Most enrichment modules have an `if __name__ == '__main__':` block at the bottom that loads `.env` and runs the function on a hard-coded test email/month. Functions as both a smoke test and a runnable example.

### Pipeline-level recommendations (small, non-alienating)
- **Standardize on one `s3_storage` helper** (e.g. shared `_lib/s3_storage.py` copied or symlinked at deploy), or accept the duplication explicitly with a one-line comment in each. The current duplication has drifted slightly across domains.
- **Replace `sys.path.insert(...)` with installable packages** via a top-level `pyproject.toml` (or one per domain). `python -m biometrics.main` then works without path hacks. Skip if it feels invasive — the existing pattern is consistent.
- **Centralize the argparse date-range setup** in a tiny `_lib/cli.py` with `add_period_args(parser)` and `resolve_period(args)`. Every `main.py` currently re-implements the same `--current-month`/`--start-date`/etc. block.
- **Use `logging` instead of `print`**, with a thin helper that still prints the human-friendly banners. Gains log levels for noisy cron runs.
- **Document the S3 schema** in a checked-in `docs/s3-schema.md` listing each path + its top-level fields. Protects the pipeline ↔ app contract from silent breakage when an enrichment renames a field.
- **Promote inline `__main__` blocks to `tests/smoke_*.py`** (or pytest-style). They already function as tests; a folder makes them discoverable and CI-runnable later.

---

## Summary

- **App half**: single Flask app, single `prod` stage, S3-only data, Google OAuth allow-list, deployed by Zappa with keep-warm on.
- **Pipeline half**: one folder per domain, `pull → enrich` two-stage flow, monthly-partitioned JSON in S3, CLI with consistent flags.
- **The contract between them is JSON files at known S3 paths — nothing more.**