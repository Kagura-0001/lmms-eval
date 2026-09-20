#!/usr/bin/env bash
set -euo pipefail
# Run from the fork root. Override the manifest to use a different storage mirror.
manifest="${BENCH_MANIFEST:-/mnt/hdfs/storage/user/lws/eval_cache/streamingbench_ovobench_omnimmi/v20260920/manifest.json}"
data_root="${LMMS_STREAMING_DATA_ROOT:?set an absolute local data directory}"
python tools/frame_cache.py prepare --manifest "$manifest" --data-root "$data_root" "$@"
