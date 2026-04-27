import logging
import os
from typing import Any

from flask import Flask, Response, jsonify, render_template

from utils.config import get_config, get_flask_secret_key
from utils.s3_utils import DATASET_PATHS, read_dataset, read_transcript_catalog, read_transcript_json, read_transcript_tei


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = get_flask_secret_key()


@app.route("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/transcripts")
def transcripts() -> str:
    return render_template("transcripts.html")


@app.route("/api/data/<dataset_name>")
def api_dataset(dataset_name: str) -> Any:
    if dataset_name not in DATASET_PATHS:
        return jsonify({"error": f"Unknown dataset: {dataset_name}"}), 404
    return jsonify(read_dataset(dataset_name))


@app.route("/api/map-data")
def api_map_data() -> Any:
    return jsonify(
        {
            "people": read_dataset("people"),
            "events": read_dataset("events"),
            "claims": read_dataset("claims"),
            "locations": read_dataset("locations"),
            "sources": read_dataset("sources"),
        }
    )


@app.route("/api/transcripts")
def api_transcripts() -> Any:
    return jsonify(read_transcript_catalog())


@app.route("/api/transcripts/<source_id>")
def api_transcript(source_id: str) -> Any:
    transcript = read_transcript_json(source_id)
    if transcript is None:
        return jsonify({"error": f"Unknown transcript: {source_id}"}), 404
    return jsonify(transcript)


@app.route("/api/transcripts/<source_id>/tei")
def api_transcript_tei(source_id: str) -> Any:
    tei = read_transcript_tei(source_id)
    if tei is None:
        return jsonify({"error": f"Unknown transcript TEI: {source_id}"}), 404
    return Response(tei, mimetype="application/xml")


@app.errorhandler(404)
def not_found(error: Exception) -> tuple[str, int]:
    logger.info("404: %s", error)
    return render_template("error.html", status_code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(error: Exception) -> tuple[str, int]:
    logger.exception("Unhandled app error: %s", error)
    return render_template("error.html", status_code=500, message="Something went wrong"), 500


if __name__ == "__main__":
    port = int(get_config("PORT", 5001))
    debug = not os.getenv("AWS_LAMBDA_FUNCTION_NAME")
    app.run(host="0.0.0.0", port=port, debug=debug)
