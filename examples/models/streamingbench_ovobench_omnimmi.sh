#!/usr/bin/env bash
set -euo pipefail
# Same entrypoint for smoke (--limit 8) and full (no --limit).
: "${LMMS_STREAMING_DATA_ROOT:?prepare the data first}"
: "${OPENAI_API_BASE:?candidate vLLM endpoint including /v1}"
: "${OPENAI_API_URL:?Qwen3.5-9B judge endpoint including /v1}"
: "${MODEL_VERSION:?served model identity, including checkpoint revision}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
python -m lmms_eval --config configs/streamingbench_ovobench_omnimmi.yaml \
  --model_args "model_version=${MODEL_VERSION},base_url=${OPENAI_API_BASE},api_key=EMPTY,num_cpus=${CONCURRENCY:-8},max_retries=2,timeout=600,is_qwen3_vl=False,prefix_aware_queue=False" \
  --output_path "${OUTPUT_PATH:-results/streamingbench_ovobench_omnimmi}" "$@"
