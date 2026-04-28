import argparse
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from enrichment.geolocate_locations import run_geolocation  # noqa: E402
from enrichment.pipeline import run_enrichment  # noqa: E402
from sources.hadc_source import pull_corpus  # noqa: E402
from utils.s3_storage import make_storage  # noqa: E402


if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(PIPELINE_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haymarket trial corpus pipeline")
    parser.add_argument("--user", default="local@example.com", help="User email for future S3 path compatibility")
    parser.add_argument("--action", choices=["pull", "enrich", "geocode", "all"], default="all")
    parser.add_argument("--corpus", choices=["test", "full"], default="test")
    parser.add_argument("--test", action="store_true", help="Shortcut for --corpus test")
    parser.add_argument("--source", choices=["hadc"], default="hadc")
    parser.add_argument("--output", choices=["local", "s3"], default="local")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--s3-bucket", default=os.getenv("HAYMARKET_S3_BUCKET", ""))
    parser.add_argument("--s3-prefix", default=os.getenv("HAYMARKET_S3_PREFIX", ""))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--clear-data", action="store_true", help="Clear generated Haymarket raw/enriched artifacts before running")
    parser.add_argument("--llm-provider", choices=["openai"], default="openai")
    parser.add_argument(
        "--llm-model",
        nargs="+",
        default=None,
        help=(
            "One or more OpenAI model names. Defaults: --test runs a comparison slate "
            "(gpt-4o-mini, gpt-4.1-mini, gpt-4o); --corpus full runs gpt-4o."
        ),
    )
    parser.add_argument(
        "--briefing-model",
        default="gpt-4.1-mini",
        help="Fixed model used for Stage A briefing (shared across model slate).",
    )
    parser.add_argument(
        "--max-tagging-workers",
        type=int,
        default=8,
        help="Concurrent OpenAI calls during Stage C per-unit tagging.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Suppress per-stage progress lines (useful in CI).",
    )
    parser.add_argument("--geocode", action="store_true", help="Run geolocation after enrichment")
    parser.add_argument("--geocoder", choices=["google"], default="google")
    parser.add_argument("--geocode-llm-model", default="gpt-4o-mini")
    parser.add_argument("--google-maps-api-key", default=os.getenv("GOOGLE_MAPS_API_KEY", ""))
    parser.add_argument("--force-geocode", action="store_true", help="Re-run geolocation for locations with Google coordinates")
    return parser.parse_args()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clear_generated_data(storage) -> dict[str, int]:
    prefixes = ["raw/haymarket", "enriched/haymarket"]
    return {prefix: storage.clear_prefix(prefix) for prefix in prefixes}


def main() -> None:
    args = parse_args()
    if args.test:
        args.corpus = "test"
    if args.llm_model is None:
        args.llm_model = (
            ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"] if args.corpus == "test" else ["gpt-4o"]
        )

    run_id = args.run_id or make_run_id()
    storage = make_storage(
        output=args.output,
        data_dir=Path(args.data_dir),
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )

    print("=" * 72)
    print("Haymarket corpus pipeline")
    print(f"Action: {args.action} | Corpus: {args.corpus} | Output: {args.output} | Run: {run_id}")
    print("=" * 72)

    try:
        if args.clear_data:
            cleared = clear_generated_data(storage)
            total = sum(cleared.values())
            print("Cleared generated Haymarket data:")
            for prefix, count in cleared.items():
                print(f"- {prefix}: {count} file(s)")
            print(f"Total cleared: {total} file(s)")

        if args.action in {"pull", "all"}:
            pages = pull_corpus(args.corpus, storage, run_id)
            print(f"Pulled {len(pages)} HADC pages")

        if args.action in {"enrich", "all"}:
            result = run_enrichment(
                storage=storage,
                run_id=run_id,
                corpus=args.corpus,
                llm_provider=args.llm_provider,
                llm_models=args.llm_model,
                briefing_model=args.briefing_model,
                max_tagging_workers=args.max_tagging_workers,
                streaming=not args.no_streaming,
            )
            print(
                "Enriched "
                f"{len(result['people'])} people, "
                f"{len(result['locations'])} locations, "
                f"{len(result['claims'])} claims, "
                f"{len(result['events'])} events"
            )
            cost_total = result["cost_summary"]["totals"]["cost_usd"]
            print(f"LLM cost for run: ${cost_total:.6f}")

        if args.action == "geocode" or (args.action in {"enrich", "all"} and args.geocode):
            geocode_result = run_geolocation(
                storage=storage,
                run_id=run_id,
                llm_provider=args.llm_provider,
                llm_model=args.geocode_llm_model,
                geocoder=args.geocoder,
                google_api_key=args.google_maps_api_key,
                force=args.force_geocode,
            )
            summary = geocode_result["summary"]
            print(
                "Geocoded "
                f"{summary['successes']} of {summary['locations']} locations "
                f"({summary['errors']} errors, {summary['skipped']} skipped), "
                f"LLM cost=${summary['cost_usd']:.6f}"
            )

        print("Pipeline completed")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
