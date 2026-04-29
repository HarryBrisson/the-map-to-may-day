#!/usr/bin/env bash
# Run a curated comparison of Stage B / Stage C configurations on a single page.
# Each experiment writes its audits to a named run_id so they sit side-by-side
# and you can inspect after the fact:
#
#   data/raw/haymarket/llm/<run_id>/<model_slug>/<page_id>/...
#   data/enriched/haymarket/model_evals/<run_id>.json
#   data/enriched/haymarket/llm_costs/<run_id>.json
#
# Usage:
#   ./experiments.sh                  # runs the default page (i019_052)
#   PAGE=x0010 ./experiments.sh       # cheap smoke test on a tiny page
#   PAGE=n017_105 ./experiments.sh    # expensive — full 113k-char page
#
# Comment out any experiment you don't want; you'll see the rough cost
# estimate next to each line.

set -euo pipefail

PAGE="${PAGE:-i019_052}"
PIPELINE=(python3 pipeline/main.py)

# ----------------------------------------------------------------------------
# Step 1: pull test corpus once. Skipped if already pulled.
# ----------------------------------------------------------------------------
if [[ ! -f data/raw/haymarket/hadc/latest_run.json ]]; then
    echo "==> No corpus on disk. Pulling test corpus (one-time)..."
    "${PIPELINE[@]}" --action pull --corpus test
else
    PULLED_RUN=$(jq -r '.run_id' data/raw/haymarket/hadc/latest_run.json)
    echo "==> Reusing pulled corpus from $PULLED_RUN"
fi

# ----------------------------------------------------------------------------
# Helper: run one experiment with a named run_id. Failures are logged but
# do NOT abort the script — we want to see how every config performs even
# when some don't validate, so the comparison summary at the end is complete.
# ----------------------------------------------------------------------------
LABELS=()
FAILURES=()
run_experiment() {
    local label="$1"; shift
    LABELS+=("$label")
    echo
    echo "================================================================"
    echo "Experiment: $label"
    echo "Args: $*"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "================================================================"
    if ! "${PIPELINE[@]}" --action enrich --pages "$PAGE" --run-id "$label" "$@"; then
        echo "  (experiment $label exited non-zero — continuing to next)"
        FAILURES+=("$label")
    fi
}

# ----------------------------------------------------------------------------
# Step 2: experiments. Comment out the ones you don't want today.
# Costs below are rough, for a 42k-char page (i019_052). Smaller pages
# cost proportionally less; n017_105 (113k) is ~3x.
# ----------------------------------------------------------------------------

# --- Cheap-tier: ~$0.70 each. Worth running for any model exploration. ---

# Does JSONL output let gpt-4.1-mini transcribe cleanly where TEI mode crashed?
run_experiment "exp_41mini_jsonl_5minitag" \
    --llm-model gpt-4.1-mini \
    --transcription-format jsonl \
    --tagging-model gpt-5-mini

# Mid-tier model with cheap tagging — does this hold up on long pages?
run_experiment "exp_54mini_jsonl_5minitag" \
    --llm-model gpt-5.4-mini \
    --transcription-format jsonl \
    --tagging-model gpt-5-mini

# --- Mid-tier: ~$1.50. Stage B uses flagship, Stage C uses cheap. ---

# Flagship transcription, cheap tagging — best fidelity at sane cost.
run_experiment "exp_55tei_5minitag" \
    --llm-model gpt-5.5 \
    --transcription-format tei \
    --tagging-model gpt-5-mini

# Same, but JSONL — direct A/B vs the row above to isolate format effect.
run_experiment "exp_55jsonl_5minitag" \
    --llm-model gpt-5.5 \
    --transcription-format jsonl \
    --tagging-model gpt-5-mini

# --- Flagship-only: ~$13. Reproduces the earlier known-good configuration. ---
# Comment back in if you want to re-baseline; otherwise skip.
# run_experiment "exp_55tei_55tag" \
#     --llm-model gpt-5.5 \
#     --transcription-format tei

# ----------------------------------------------------------------------------
# Tagging-focused experiments: hold Stage B constant, vary Stage C.
# All use gpt-5.4-mini + JSONL for transcription so the per-run Stage B
# output is roughly comparable. Variance: ~5-10% in entity counts will be
# Stage B noise, not Stage C signal — anything bigger is the tagging model.
# Total tagging-focused cost: ~$1.50 across the three.
# ----------------------------------------------------------------------------

# Cheapest tagging — does nano produce decent entities on per-<sp> chunks?
run_experiment "exp_tag_5nano" \
    --llm-model gpt-5.4-mini \
    --transcription-format jsonl \
    --tagging-model gpt-5-nano

# Mid-cheap — gpt-5-mini is the baseline used in the experiments above.
run_experiment "exp_tag_5mini" \
    --llm-model gpt-5.4-mini \
    --transcription-format jsonl \
    --tagging-model gpt-5-mini

# Mid-priced — does upgrading tagging meaningfully beat 5-mini, or is it
# the same with extra cost?
run_experiment "exp_tag_54mini" \
    --llm-model gpt-5.4-mini \
    --transcription-format jsonl \
    --tagging-model gpt-5.4-mini

# ----------------------------------------------------------------------------
# Step 3: print a quick comparison from the model_evals.
# ----------------------------------------------------------------------------
echo
echo "================================================================"
echo "All experiments complete."
if (( ${#FAILURES[@]} > 0 )); then
    echo "Failed (Stage B validation or other error): ${FAILURES[*]}"
    echo "Their audits are still on disk under data/raw/haymarket/llm/<label>/"
fi
echo "================================================================"
echo
echo "Audit dirs:"
for label in "${LABELS[@]}"; do
    printf "  data/raw/haymarket/llm/%s/\n" "$label"
done

echo
echo "Quick comparison (per experiment):"
for label in "${LABELS[@]}"; do
    eval_path="data/enriched/haymarket/model_evals/${label}.json"
    if [[ -f "$eval_path" ]]; then
        echo
        echo "  $label"
        jq -r '.models[] | "    model=\(.model) people=\(.people_count) locs=\(.location_count) claims=\(.claim_count) events=\(.event_suggestion_count) quotes=\(.quote_count) tei_valid=\(.tei_valid_count)/\(.pages) cost=$\(.cost_usd)"' \
            "$eval_path"
    fi
done

echo
echo "To inspect a specific failed unit:"
echo "  jq -r 'select(.status==\"error\") | \"\\(.unit_id)\\t\\(.error)\"' data/raw/haymarket/llm/<label>/<model>/source_hadc_${PAGE}/tagging.jsonl"
