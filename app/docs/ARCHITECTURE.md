# App Architecture

The app is a small Flask/Zappa dashboard that reads app-ready Haymarket JSON artifacts from a configured backend.

## Runtime

- `app.py` owns routes and the Flask app object.
- Local development runs on port `5001`.
- `/health` is the keep-warm and smoke-test endpoint.
- `utils/config.py` resolves config from env vars, `creds.json`, then defaults.
- `utils/s3_utils.py` reads JSON from local files by default and S3 when configured.

## Frontend

- `templates/index.html` hosts the map shell and filters.
- `static/js/common/api.js` owns API fetches.
- `static/js/pages/map.js` renders Leaflet markers, filters, person colors, event cards, and source-linked claim details.
- `static/css/styles.css` contains page styling.
