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
from enrichment.brief_harmonization import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ESCALATION_REVIEW_MODEL,
    DEFAULT_MAX_LLM_REVIEW_BATCHES,
    DEFAULT_REVIEW_MODEL,
)
from enrichment.pipeline import run_brief_harmonization_update, run_brief_update, run_enrichment  # noqa: E402
from sources.hadc_source import pull_corpus  # noqa: E402
from utils.s3_storage import make_storage  # noqa: E402


if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(PIPELINE_ROOT / ".env")


MODEL_SLATES: dict[str, list[str]] = {
    # Two cheapest options — fastest iteration on a small page.
    "small": ["gpt-4.1-mini", "gpt-5-mini"],
    # Cheap tier — under ~$0.50/1M input. Good baseline for cost/quality.
    "cheap": ["gpt-4.1-mini", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-nano"],
    # Mid tier — punchier than cheap, still affordable.
    "mid": ["gpt-5.4-mini", "gpt-4.1", "gpt-5"],
    # Flagship tier — highest fidelity, highest cost.
    "flagship": ["gpt-5.5", "gpt-5.4"],
    # Compare across tiers — one cheap, one mid, one flagship.
    "compare": ["gpt-4.1-mini", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.5"],
    # Everything we have pricing for. Expensive — usually not what you want.
    "all": [
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.4-mini",
        "gpt-5.4",
        "gpt-5.5",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haymarket trial corpus pipeline")
    parser.add_argument("--user", default="local@example.com", help="User email for future S3 path compatibility")
    parser.add_argument(
        "--action",
        choices=["pull", "briefs", "harmonize-briefs", "enrich", "geocode", "all", "cache-status"],
        default="all",
    )
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
            "(gpt-4o-mini, gpt-4.1-mini, gpt-4o); --corpus full runs gpt-4o. "
            "Can be combined with --slate to merge a preset list."
        ),
    )
    parser.add_argument(
        "--slate",
        choices=sorted(MODEL_SLATES.keys()),
        default=None,
        help=(
            "Run a predefined comparison slate. Sets --llm-model to a curated list of "
            "models so you can A/B without remembering names. Overrides --llm-model unless "
            "you also pass --llm-model (then the two are merged, deduped)."
        ),
    )
    parser.add_argument(
        "--briefing-model",
        default="gpt-5-mini",
        help=(
            "Fixed model used for Stage A briefing (shared across model slate). "
            "Default gpt-5-mini: reasoning-class but cheaper than gpt-4.1-mini and "
            "produces more consistent speaker_directory IDs/names than non-reasoning "
            "models."
        ),
    )
    parser.add_argument(
        "--tagging-model",
        default=None,
        help=(
            "Optional fixed model for Stage C tagging. If unset, the per-slate --llm-model is "
            "used. Recommend a cheaper model here (e.g. gpt-5-mini, gpt-4.1-nano) since each "
            "tagging call is small and structured."
        ),
    )
    parser.add_argument(
        "--max-tagging-workers",
        type=int,
        default=8,
        help="Concurrent OpenAI calls during Stage C per-unit tagging.",
    )
    parser.add_argument(
        "--briefing-attempts",
        type=int,
        default=3,
        help=(
            "Max attempts for Stage A briefing per source. The first attempt counts; "
            "additional attempts only fire when the prior attempt errors. Default 3."
        ),
    )
    parser.add_argument(
        "--max-brief-workers",
        type=int,
        default=1,
        help=(
            "Concurrent OpenAI calls for --action briefs. Default 1 preserves sequential output; "
            "Tier 5 users can usually try 32 or 64 with resume enabled."
        ),
    )
    parser.add_argument(
        "--tagging-attempts",
        type=int,
        default=3,
        help=(
            "Max attempts per tagging unit. The first attempt counts; subsequent attempts "
            "only fire when the prior attempt errored. Default 3."
        ),
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Suppress per-stage progress lines (useful in CI).",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        default=None,
        help=(
            "Restrict enrichment to these page IDs (e.g. --pages source_hadc_x0010). "
            "Supports prefix match — 'x0010' matches 'source_hadc_x0010'. The pull "
            "phase still fetches the full --corpus, only enrichment is filtered."
        ),
    )
    parser.add_argument(
        "--transcription-attempts",
        type=int,
        default=2,
        help=(
            "Max attempts for Stage B transcription per page. The first attempt counts; "
            "additional attempts only fire on validation failure. Default 2."
        ),
    )
    parser.add_argument(
        "--transcription-format",
        choices=["tei", "jsonl"],
        default="tei",
        help=(
            "Stage B output format. 'tei' asks the model for raw TEI XML (default). 'jsonl' asks "
            "for one JSON object per source line — the pipeline then synthesizes well-formed TEI "
            "from the rows. JSONL is more robust against XML well-formedness mistakes."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Skip the page-level bundle cache. Default: cached bundles for pages with the same "
            "config (models, prompts, source-text hash) are reused, and successful new bundles "
            "are written to data/cache/haymarket/. Pass --no-cache to force every page to re-run "
            "all three stages and not write back to the cache. For --action briefs, this also "
            "skips the durable brief cache."
        ),
    )
    parser.add_argument(
        "--no-resume-briefs",
        action="store_true",
        help=(
            "For --action briefs, ignore successful briefing audit files already written under "
            "the same --run-id and call the model again. By default, brief updates resume and "
            "reuse successful per-page briefs before checking the durable brief cache."
        ),
    )
    parser.add_argument(
        "--fresh-briefs",
        action="store_true",
        help=(
            "For --action briefs, ignore same-run resume files and the durable brief cache, "
            "call the model again, and do not overwrite matching cached briefs."
        ),
    )
    parser.add_argument(
        "--no-brief-harmonization",
        action="store_true",
        help=(
            "For --action briefs, write raw source navigation directly from briefs and skip the "
            "Stage A.5 harmonized brief artifact."
        ),
    )
    parser.add_argument(
        "--brief-harmonization-embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Embedding model for Stage A.5 harmonization. Default text-embedding-3-large. "
            "Use text-embedding-3-small for budget exploratory runs."
        ),
    )
    parser.add_argument(
        "--brief-harmonization-embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help=(
            "Embedding dimensions for Stage A.5 harmonization. Default 1024 for "
            "text-embedding-3-large; use 512 with text-embedding-3-small for budget mode."
        ),
    )
    parser.add_argument(
        "--no-brief-harmonization-embeddings",
        action="store_true",
        help="Skip semantic embedding generation during Stage A.5 harmonization.",
    )
    parser.add_argument(
        "--brief-harmonization-review-model",
        default=DEFAULT_REVIEW_MODEL,
        help="Structured-output model for ambiguous Stage A.5 merge/classification review.",
    )
    parser.add_argument(
        "--brief-harmonization-escalation-model",
        default=DEFAULT_ESCALATION_REVIEW_MODEL,
        help=(
            "Structured-output model for high-risk Stage A.5 review batches. "
            "Default matches --brief-harmonization-review-model; pass gpt-5.5 explicitly "
            "only when you want expensive escalation."
        ),
    )
    parser.add_argument(
        "--no-brief-harmonization-llm-review",
        action="store_true",
        help="Skip LLM confirmation for ambiguous Stage A.5 candidate matches.",
    )
    parser.add_argument(
        "--brief-harmonization-max-review-batches",
        type=int,
        default=DEFAULT_MAX_LLM_REVIEW_BATCHES,
        help=(
            "Maximum new Stage A.5 LLM review batches to run before stopping review work. "
            "Completed batch artifacts are reused and do not count against the cap. "
            "Use -1 for no cap. Default 4."
        ),
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Wipe durable Haymarket bundle and brief caches before running (separate from --clear-data).",
    )
    parser.add_argument("--geocode", action="store_true", help="Run geolocation after enrichment")
    parser.add_argument("--geocoder", choices=["google"], default="google")
    parser.add_argument("--geocode-llm-model", default="gpt-4o-mini")
    parser.add_argument("--google-maps-api-key", default=os.getenv("GOOGLE_MAPS_API_KEY", ""))
    parser.add_argument("--force-geocode", action="store_true", help="Re-run geolocation for locations with Google coordinates")
    return parser.parse_args()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def print_cache_status(data_dir: str) -> None:
    from utils.page_cache import cache_inventory

    inventory = cache_inventory(data_dir)
    base = inventory["base_path"]
    if inventory["total_pages"] == 0:
        print(f"Cache empty ({base})")
        return

    print(f"Cache: {base}")
    print(
        f"  {len(inventory['configs'])} bundle config(s), "
        f"{len(inventory.get('brief_configs', []))} brief config(s), "
        f"{inventory['total_pages']} cached page(s), "
        f"${inventory['total_cost_usd']:.4f} of producer cost"
    )
    for cfg in inventory["configs"]:
        config = cfg["config"] or {}
        config_summary = (
            f"briefing={config.get('briefing_model', '?')}  "
            f"transcription={config.get('transcription_model', '?')}/{config.get('transcription_format', '?')}  "
            f"tagging={config.get('tagging_model', '?')}"
        )
        print(
            f"\n  config {cfg['cache_key']}  "
            f"({cfg['page_count']} page(s), ${cfg['cost_usd']:.4f})"
        )
        print(f"    {config_summary}")
        for entry in cfg["pages"]:
            produced = (entry.get("produced_at") or "")[:19].replace("T", " ")
            run_label = entry.get("produced_by_run_id") or "?"
            print(
                f"    - {entry['page_id']:35s}  ${entry['cost_usd']:>7.4f}  "
                f"{produced}  by {run_label}"
            )
    for cfg in inventory.get("brief_configs", []):
        config = cfg["config"] or {}
        print(
            f"\n  brief config {cfg['cache_key']}  "
            f"({cfg['page_count']} page(s), ${cfg['cost_usd']:.4f})"
        )
        print(
            f"    briefing={config.get('briefing_model', '?')}  "
            f"prompt={config.get('briefing_prompt_template', '?')}  "
            f"schema={config.get('briefing_schema_digest', '?')}"
        )
        for entry in cfg["pages"]:
            produced = (entry.get("produced_at") or "")[:19].replace("T", " ")
            run_label = entry.get("produced_by_run_id") or "?"
            print(
                f"    - {entry['page_id']:35s}  ${entry['cost_usd']:>7.4f}  "
                f"{produced}  by {run_label}"
            )


def clear_generated_data(storage) -> dict[str, int]:
    prefixes = ["raw/haymarket", "enriched/haymarket"]
    return {prefix: storage.clear_prefix(prefix) for prefix in prefixes}


def main() -> None:
    args = parse_args()
    if args.test:
        args.corpus = "test"
    if (
        args.brief_harmonization_embedding_model == "text-embedding-3-small"
        and args.brief_harmonization_embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    ):
        args.brief_harmonization_embedding_dimensions = 512
    max_harmonization_review_batches = (
        None
        if args.brief_harmonization_max_review_batches is not None
        and args.brief_harmonization_max_review_batches < 0
        else args.brief_harmonization_max_review_batches
    )
    if args.slate:
        slate_models = MODEL_SLATES[args.slate]
        if args.llm_model:
            # Merge user-provided models with slate, dedupe preserving order
            seen: set[str] = set()
            merged: list[str] = []
            for model in slate_models + list(args.llm_model):
                if model not in seen:
                    seen.add(model)
                    merged.append(model)
            args.llm_model = merged
        else:
            args.llm_model = list(slate_models)
    elif args.llm_model is None:
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
        if args.clear_cache:
            from utils.page_cache import clear_cache as clear_page_cache

            cleared_count = clear_page_cache(storage)
            print(f"Cleared page cache: {cleared_count} file(s) removed from cache/haymarket/")

        if args.clear_data:
            cleared = clear_generated_data(storage)
            total = sum(cleared.values())
            print("Cleared generated Haymarket data:")
            for prefix, count in cleared.items():
                print(f"- {prefix}: {count} file(s)")
            print(f"Total cleared: {total} file(s)")

        if args.action == "cache-status":
            print_cache_status(args.data_dir)
            return

        if args.action in {"pull", "all"}:
            pages = pull_corpus(args.corpus, storage, run_id)
            print(f"Pulled {len(pages)} HADC pages")

        if args.action == "briefs":
            result = run_brief_update(
                storage=storage,
                run_id=run_id,
                corpus=args.corpus,
                briefing_model=args.briefing_model,
                streaming=not args.no_streaming,
                page_filter=args.pages,
                resume=(not args.no_resume_briefs and not args.fresh_briefs),
                max_brief_workers=args.max_brief_workers,
                max_briefing_attempts=args.briefing_attempts,
                use_brief_cache=(not args.no_cache and not args.fresh_briefs),
                write_brief_cache=(not args.no_cache and not args.fresh_briefs),
                harmonize_briefs=not args.no_brief_harmonization,
                brief_harmonization_use_embeddings=not args.no_brief_harmonization_embeddings,
                brief_harmonization_embedding_model=args.brief_harmonization_embedding_model,
                brief_harmonization_embedding_dimensions=args.brief_harmonization_embedding_dimensions,
                brief_harmonization_use_llm_review=not args.no_brief_harmonization_llm_review,
                brief_harmonization_review_model=args.brief_harmonization_review_model,
                brief_harmonization_escalation_model=args.brief_harmonization_escalation_model,
                brief_harmonization_max_review_batches=max_harmonization_review_batches,
            )
            harmonization = result.get("harmonization") or {}
            coverage = harmonization.get("coverage") or {}
            print(
                "Updated source briefs: "
                f"{result['briefings']} of {result['pages']} page(s) briefed, "
                f"{result['reused']} reused, "
                f"{result['cache_hits']} durable cache hit(s), "
                f"{result['skipped']} skipped, "
                f"{len(result['sources'])} source navigation records written"
            )
            if harmonization:
                harmonization_cost = (
                    float((harmonization.get("embedding_cache_manifest") or {}).get("cost_usd") or 0.0)
                    + float(((harmonization.get("qa_report") or {}).get("llm_review") or {}).get("cost_usd") or 0.0)
                )
                print(
                    "Brief harmonization: "
                    f"{coverage.get('harmonized_sources', 0)} source(s), "
                    f"{coverage.get('event_clusters', 0)} event cluster(s), "
                    f"{coverage.get('document_event_clusters', 0)} document-event cluster(s), "
                    f"coverage_ok={coverage.get('coverage_ok')}"
                )
                print(f"Brief harmonization cost: ${harmonization_cost:.6f}")
            print(f"LLM cost for brief update: ${result['cost_usd']:.6f}")

        if args.action == "harmonize-briefs":
            result = run_brief_harmonization_update(
                storage=storage,
                run_id=run_id,
                corpus=args.corpus,
                briefing_model=args.briefing_model,
                page_filter=args.pages,
                brief_harmonization_use_embeddings=not args.no_brief_harmonization_embeddings,
                brief_harmonization_embedding_model=args.brief_harmonization_embedding_model,
                brief_harmonization_embedding_dimensions=args.brief_harmonization_embedding_dimensions,
                brief_harmonization_use_llm_review=not args.no_brief_harmonization_llm_review,
                brief_harmonization_review_model=args.brief_harmonization_review_model,
                brief_harmonization_escalation_model=args.brief_harmonization_escalation_model,
                brief_harmonization_max_review_batches=max_harmonization_review_batches,
            )
            coverage = (result.get("harmonization") or {}).get("coverage") or {}
            print(
                "Harmonized source briefs: "
                f"{result['briefings']} of {result['pages']} page(s), "
                f"{result['skipped']} skipped, "
                f"{len(result['sources'])} source navigation records written"
            )
            print(
                "Brief harmonization: "
                f"{coverage.get('event_clusters', 0)} event cluster(s), "
                f"{coverage.get('document_event_clusters', 0)} document-event cluster(s), "
                f"coverage_ok={coverage.get('coverage_ok')}"
            )
            harmonization = result.get("harmonization") or {}
            harmonization_cost = (
                float((harmonization.get("embedding_cache_manifest") or {}).get("cost_usd") or 0.0)
                + float(((harmonization.get("qa_report") or {}).get("llm_review") or {}).get("cost_usd") or 0.0)
            )
            print(f"Brief harmonization cost: ${harmonization_cost:.6f}")

        if args.action in {"enrich", "all"}:
            result = run_enrichment(
                storage=storage,
                run_id=run_id,
                corpus=args.corpus,
                llm_provider=args.llm_provider,
                llm_models=args.llm_model,
                briefing_model=args.briefing_model,
                tagging_model=args.tagging_model,
                max_tagging_workers=args.max_tagging_workers,
                max_tagging_unit_attempts=args.tagging_attempts,
                streaming=not args.no_streaming,
                page_filter=args.pages,
                max_briefing_attempts=args.briefing_attempts,
                max_transcription_attempts=args.transcription_attempts,
                transcription_format=args.transcription_format,
                use_cache=not args.no_cache,
                brief_harmonization_use_embeddings=not args.no_brief_harmonization_embeddings,
                brief_harmonization_embedding_model=args.brief_harmonization_embedding_model,
                brief_harmonization_embedding_dimensions=args.brief_harmonization_embedding_dimensions,
                brief_harmonization_use_llm_review=not args.no_brief_harmonization_llm_review,
                brief_harmonization_review_model=args.brief_harmonization_review_model,
                brief_harmonization_escalation_model=args.brief_harmonization_escalation_model,
                brief_harmonization_max_review_batches=max_harmonization_review_batches,
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
