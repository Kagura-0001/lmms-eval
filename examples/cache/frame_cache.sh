#!/usr/bin/env bash
set -euo pipefail
# Run from the fork root. Prepare data and explicitly select its manifest.
manifest="${BENCH_MANIFEST:-}"
: "${manifest:?set BENCH_MANIFEST to the manifest for your prepared data}"
data_root="${LMMS_STREAMING_DATA_ROOT:?set an absolute local data directory}"
python tools/frame_cache.py prepare --manifest "$manifest" --data-root "$data_root" "$@"
