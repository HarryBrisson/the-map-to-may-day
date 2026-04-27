# App Styleguide

The app should feel like a research dashboard for historical evidence: readable, restrained, and source-forward.

## UI Principles

- Keep filters visible beside the map.
- Show source links and quoted evidence close to each event.
- Use color to distinguish people, not to imply certainty.
- Surface confidence and missing coordinates as data quality signals.

## JavaScript

- Use vanilla JavaScript modules/files under `static/js/pages` and `static/js/common`.
- Keep API calls in `static/js/common/api.js`.
- Keep page-specific state in the page script.
