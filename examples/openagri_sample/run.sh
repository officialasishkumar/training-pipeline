#!/usr/bin/env bash
# End-to-end walkthrough: 12 synthetic logs -> seeds -> generated trajectories
# -> tag -> validate -> stratify -> persona score -> SFT + DPO export.
#
# Uses the *stub* LLM backend by default so the whole thing finishes in well
# under 30 s on a laptop with no GPU. Swap in a real model with:
#
#   GENERATE_BACKEND=transformers GENERATE_MODEL=Qwen/Qwen2.5-7B-Instruct ./run.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"

# Pick the tp executable. Prefer the project-local venv if one exists.
if [ -x "$ROOT/.venv/bin/tp" ]; then
  TP="$ROOT/.venv/bin/tp"
elif command -v tp >/dev/null 2>&1; then
  TP="tp"
else
  echo "tp not found on PATH and no .venv at $ROOT/.venv. Run: pip install -e .[dev] from the repo root." >&2
  exit 1
fi

GENERATE_BACKEND="${GENERATE_BACKEND:-stub}"
GENERATE_MODEL="${GENERATE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "$BUILD"
rm -f "$BUILD"/*.jsonl
rm -rf "$BUILD/sft" "$BUILD/dpo"

echo "==> 1/8 ingest"
"$TP" ingest \
  --input  "$HERE/logs.jsonl" \
  --output "$BUILD/canonical.jsonl"

echo "==> 2/8 redact (PII orchestrator + leakage gate)"
"$TP" redact \
  --input  "$BUILD/canonical.jsonl" \
  --output "$BUILD/redacted.jsonl" \
  --quarantine "$BUILD/redaction_quarantine.jsonl"

echo "==> 3/8 generate seeds (cluster intents)"
"$TP" generate seeds \
  --input  "$BUILD/redacted.jsonl" \
  --output "$BUILD/seeds.jsonl" \
  --similarity-threshold 0.55

echo "==> 4/8 generate trajectories (backend: $GENERATE_BACKEND)"
"$TP" generate trajectories \
  --seeds  "$BUILD/seeds.jsonl" \
  --output "$BUILD/synthetic.jsonl" \
  --tool-registry "$HERE/tools.yaml" \
  --backend "$GENERATE_BACKEND" \
  --model   "$GENERATE_MODEL" \
  --max-steps 5

echo "==> 5/8 tag complexity / recovery / ambiguity"
"$TP" tag \
  --input  "$BUILD/synthetic.jsonl" \
  --output "$BUILD/tagged.jsonl"

echo "==> 6/8 validate tool-call/observation consistency"
"$TP" validate \
  --input  "$BUILD/tagged.jsonl" \
  --tool-registry "$HERE/tools.yaml" \
  --output "$BUILD/validated.jsonl"

echo "==> 7/8 stratify (cap per (difficulty x edge-case) bucket)"
"$TP" generate stratify \
  --input  "$BUILD/validated.jsonl" \
  --output "$BUILD/stratified.jsonl" \
  --cap-per-bucket 5

echo "==> 7b/8 persona score"
"$TP" score \
  --persona "$HERE/persona.md" \
  --input   "$BUILD/stratified.jsonl" \
  --output  "$BUILD/scored.jsonl"

echo "==> 8/8 export SFT + DPO"
"$TP" export sft \
  --input "$BUILD/scored.jsonl" \
  --output-dir "$BUILD/sft" \
  --template chatml \
  --shard-size 1000

"$TP" export dpo \
  --input "$BUILD/scored.jsonl" \
  --output-dir "$BUILD/dpo" \
  --strategy all \
  --persona "$HERE/persona.md"

echo
echo "================ Pipeline stats ================"

count_jsonl() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "0"
    return
  fi
  awk 'NF' "$path" | wc -l | tr -d ' '
}

count_glob() {
  local glob="$1"
  local total=0
  for f in $glob; do
    [ -f "$f" ] || continue
    total=$((total + $(awk 'NF' "$f" | wc -l)))
  done
  echo "$total"
}

printf "%-40s %s\n" "input log trajectories"            "$(count_jsonl "$HERE/logs.jsonl")"
printf "%-40s %s\n" "redacted trajectories"             "$(count_jsonl "$BUILD/redacted.jsonl")"
printf "%-40s %s\n" "seeds (clustered intents)"         "$(count_jsonl "$BUILD/seeds.jsonl")"
printf "%-40s %s\n" "synthetic trajectories"            "$(count_jsonl "$BUILD/synthetic.jsonl")"
printf "%-40s %s\n" "tagged"                            "$(count_jsonl "$BUILD/tagged.jsonl")"
printf "%-40s %s\n" "validated"                         "$(count_jsonl "$BUILD/validated.jsonl")"
printf "%-40s %s\n" "stratified (capped per bucket)"    "$(count_jsonl "$BUILD/stratified.jsonl")"
printf "%-40s %s\n" "scored"                            "$(count_jsonl "$BUILD/scored.jsonl")"
printf "%-40s %s\n" "SFT records"                       "$(count_glob "$BUILD/sft/sft-*.jsonl")"
printf "%-40s %s\n" "DPO pairs"                         "$(count_glob "$BUILD/dpo/dpo-*.jsonl")"

echo
echo "PII findings by category (post-redaction summary):"
if [ -f "$BUILD/redaction_quarantine.jsonl" ]; then
  awk 'NF' "$BUILD/redaction_quarantine.jsonl" \
    | python3 -c "
import json, sys
counts = {}
for line in sys.stdin:
    rec = json.loads(line)
    for leak in rec.get('leaks', []):
        counts[leak['category']] = counts.get(leak['category'], 0) + 1
if not counts:
    print('  (no leaks)')
else:
    for cat, n in sorted(counts.items()):
        print(f'  {cat:20s} {n}')
"
fi

echo
echo "Done. Artifacts in $BUILD/"
