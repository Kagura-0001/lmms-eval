# StreamingBench + OVO-Bench + OmniMMI

One native task group, `streamingbench_ovobench_omnimmi`, contains thirteen
subtasks and 9,175 questions. The original ten Causal Suite entries contain
7,485 independent questions. SG has 300 sessions/704 questions, MD 300/786,
and SI 200/200. No combined score is defined across these protocols.

This integration preserves our frozen local annotations and scoring. It does
**not** claim to reproduce AURA's complete input protocol or paper scores.
Existing upstream `ovobench`, `ovo_backward`, `ovo_realtime`, and `ovo_forward`
tasks retain their original definitions.

## Prepare on a new node

Use `uv sync` to install the fork, then activate `.venv`. Candidate and judge
vLLM servers can run in their own environments on other nodes.

```bash
source .venv/bin/activate
export LMMS_STREAMING_DATA_ROOT=/local/scratch/streaming-bench
bash examples/cache/frame_cache.sh --limit 8 --workers 8
```

The default manifest is the frozen internal snapshot at
`/mnt/hdfs/storage/user/lws/eval_cache/streamingbench_ovobench_omnimmi/v20260920/manifest.json`.
The new node needs access to that shared storage and the source media mounts.
This is **not** a newly published public Hugging Face dataset. A standalone
clone outside that environment requires an accessible mirror via
`BENCH_MANIFEST`. No credentials or private data are committed to the fork.

The fetcher accepts mounted paths, HTTP(S), `hdfs://` and Hugging Face dataset
URIs. The manifest describes annotation URLs/hashes, video URLs/sizes and
optional frame-pack URLs. It does not depend on another code checkout.
Mirror the artifacts and update only these locations to move between storage
systems; stable media keys and frozen annotation hashes remain unchanged.

Preparation fetches all annotations for selected tasks, then only the media
needed by `--limit` (per task, per session for SG/MD). Omit `--limit` to prepare
the full group. `--tasks omnimmi_md,streamingbench_rt` selects subtasks.
`--media none` fetches annotations only; `--media videos` fetches original
videos; default `--media frames` copies existing packs or explicitly builds
missing ones from downloaded videos. `prepare_receipt.json` records reuse,
copies, builds and failures. Mounted/HTTP partial transfers are resumable;
HDFS CLI transfers restart their incomplete file.

The existing 2 FPS/original-resolution/JPEG80 master covers the ten causal
entries. SG/MD/SI add 521 distinct source videos, which require their own packs.
Building these is deliberate preparation work, never a hidden inference fallback.
Cache sampling density does not change the benchmark's 1 FPS evaluation grid.

```bash
python tools/frame_cache.py inspect --data-root "$LMMS_STREAMING_DATA_ROOT"
```

Indexes and sizes are checked once on first use; large video/frame files are
not repeatedly hashed. Timestamps must exist exactly in the index. The reader
never snaps to a later frame. Packs contain original-resolution JPEG bytes;
pixel budgets are applied by the candidate server's multimodal processor.

## Evaluate

```bash
export MODEL_VERSION=qwen-step8816-7baa9a18
export OPENAI_API_BASE=http://candidate-host:8000/v1
export OPENAI_API_URL=http://judge-host:8001/v1
export OPENAI_API_KEY=EMPTY
export OUTPUT_PATH=results/qwen8816-packed-smoke
bash examples/models/streamingbench_ovobench_omnimmi.sh --limit 8

export LMMS_STREAMING_HISTORY_LAYOUT=per_frame_silence
export OUTPUT_PATH=results/qwen8816-silence-smoke
bash examples/models/streamingbench_ovobench_omnimmi.sh --limit 8
```

Use the identical entrypoint without `--limit` for full evaluation, after full
media preparation. Each task can also be selected with `--tasks <subtask>`.
`--limit 8` is eight **sessions** for SG/MD, including all of their rounds.
Use a fresh output directory for each treatment.

Candidate example: Qwen step8816 revision
`7baa9a18ac8bb54fa18158e8ff67b3773d89fd49`; serve with context 163840,
`--limit-mm-per-prompt '{"image":360}'`, and suitable GPU memory/concurrency.
The native per-task generation configuration passes `mm_processor_kwargs`:
min4096/max300000 pixels for the causal entries and max602112 for SG/MD/SI.
The server must support these request-level overrides. `num_cpus` in the
existing `async_openai` backend controls concurrent requests (not decoder CPUs).
Use CLI `--model_args` to change concurrency and endpoint.

Judge: `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, served under that model name;
128 tokens, greedy, `enable_thinking=False`. Candidate and judge endpoints are
independent. The existing judge provider is used with the original semantic
prompts, not its generic binary prompt. Missing/failed judgments remain
unscored (`NaN` plus `judge_failures`); they are not zero-score answers.

## Input and metric contracts

- Both layouts use one-decimal timestamps and no newline after images.
- `packed`: one user message containing all frames and the final question.
- `per_frame_silence`: each preceding frame has an assistant `<silence>` turn;
  the question occurs only with the final frame. These turns are supplied
  history, not extra model calls. No silence logit bias is used.
- The step8816 tokenizer registers `<silence>` as token151669. AURA's
  `<|silent|>` is a different spelling and is not substituted. This confirms
  the exported token contract, not complete Swift training-input alignment.
- Causal entries preserve full permitted prefixes, half-open ranges, 1 FPS,
  uniform max360 frames, and original query text. SQA retains prior gold QA.
  AP queries at video duration; OSU remains visual-only.
- SG/MD/SI preserve last at-most30 one-second segments and original query
  arithmetic. SG/MD run session rounds in order. MD substitutes the immediately
  preceding prediction (empty after failure), never the GT. No earlier QA is
  implicitly appended. Judges see the original question containing the marker.
- Out-of-duration annotations remain failures; there is no silent clamping,
  future-frame fallback or removal from denominators.
- OVO preserves source matching and task/mode macro averaging. StreamingBench
  preserves the existing MCQ parser. SG/MD accuracy means all answers in a
  session are correct; score averages first truncate each rating, average
  within each session, and then across sessions.
- Native `item_accuracy[index]` and `cumulative_item_accuracy[index]` are
  flattened **only at the lmms-eval metric boundary** to `item_accuracy_1`,
  `cumulative_item_accuracy_1`, etc. SG's fourth-round metric is retained.

Response caching uses native `--use_cache`, separately from frame caching.
Resolved layout/frame settings and annotation hashes are included in the task
config fingerprint and dataset documents. Multi-round responses are cached as
complete sessions. The native cache can also retain failure markers: use a
fresh cache directory when retrying failed sessions; automatic failure eviction
is not implemented here.
Always identify the actual checkpoint in `model_version` and use separate
cache roots for separately deployed judge revisions.

## Validation and boundaries

On 2026-09-20, 49 focused/existing protocol tests and the changed-file Ruff
hooks passed. Native CLI smoke covered all thirteen tasks (130 questions per
layout) with Qwen step8816 and Qwen3.5-9B judging. Preparation in two empty
node directories exercised existing-pack copying and missing-pack building;
the larger preparation staged 49 videos without failures. A same-config
response-cache replay hit all thirteen requests. These are smoke checks, not
full-dataset scores. Nine SG/MD questions had out-of-duration source timestamps;
the silence treatment additionally produced four empty AP answers.

Focused tests cover relocation, partial-copy resume, exact timestamps,
unchanged image data URLs, MD prediction isolation, query cutoffs, concurrent
sessions and native score denominators. Real CLI smoke evidence is recorded
separately from unit tests. No changes to native vLLM or OVStreaming are needed.
PA/PO and AURA's full video-input reproduction are outside this integration.

`tools/prepare_streamingbench_ovobench_omnimmi.py` is the one-time export tool
for existing frozen snapshots. Ordinary users only need `frame_cache.py` and
the manifest. Source files and historical predictions remain unchanged.
