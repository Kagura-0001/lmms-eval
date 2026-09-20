"""Frozen local protocols, expressed through native lmms-eval task interfaces."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict

from lmms_eval.api.task import ConfigurableMessagesTask
from lmms_eval.models.model_utils.packed_frames import image_content, pack_index, read_frames


class StreamingBenchmarkTask(ConfigurableMessagesTask):
    """Load a relocated frozen annotation snapshot, like other local-media tasks."""

    def __init__(self, config: dict, **kwargs) -> None:
        config = dict(config)
        config.pop("class", None)
        super().__init__(config=config, **kwargs)

    def download(self, dataset_kwargs: dict | None = None) -> None:
        """Resolve local roots before constructing the standard HF Dataset."""
        root = Path(os.environ.get("LMMS_STREAMING_DATA_ROOT", "data/streamingbench_ovobench_omnimmi")).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        name = self.config.task
        artifact = manifest["tasks"][name]
        blob = (root / "annotations" / f"{name}.jsonl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Frozen annotation mismatch: {name}")
        rows = [json.loads(line) for line in blob.splitlines() if line]
        if len(rows) != artifact["count"]:
            raise ValueError(f"Frozen annotation count mismatch: {name}")
        options = dict((self.config.lmms_eval_specific_kwargs or {}).get("default", {}))
        layout = os.environ.get("LMMS_STREAMING_HISTORY_LAYOUT", options.get("history_layout", "packed"))
        if layout not in {"packed", "per_frame_silence"}:
            raise ValueError("history_layout must be packed or per_frame_silence")
        options.update(history_layout=layout, annotation_sha256=artifact["sha256"])
        # Native response caching fingerprints dump_config(), not dataset rows.
        # Resolve environment overrides into the task config before it is hashed.
        self.config.lmms_eval_specific_kwargs = {"default": options}
        frame_root = Path(os.environ.get("LMMS_STREAMING_FRAME_CACHE", str(root / "frames"))).expanduser().resolve()
        # Persist resolved input settings in documents: sample logs and response
        # cache fingerprints must distinguish both layouts and frame budgets.
        for row in rows:
            row["profile"] = {**options, "frame_root": str(frame_root)}
        self.dataset = DatasetDict({self.config.test_split: Dataset.from_list(rows)})
        self.dataset_no_image = self.dataset.copy()


def failed(prediction: str) -> bool:
    """Recognize explicit transport/budget failure records, not valid answers."""
    return not prediction or prediction.startswith(("[LMMS_EVAL_ERROR]", "[LMMS_EVAL_BUDGET_EXCEEDED]", "[LMMS_EVAL_REQUEST_FAILED"))


def frame_times(qa: dict, max_frames: int, fps: float = 1.0) -> list[float]:
    """Preserve half-open video ranges, query cutoffs and uniform retention."""
    start, end = map(float, qa["range_sec"])
    at = float(qa["at_sec"])
    if not all(math.isfinite(value) for value in (start, end, at, fps)) or fps <= 0 or not 0 <= start <= at <= end or start >= end or max_frames < 2:
        raise ValueError("Invalid causal frame range/budget")
    times = [start + index / fps for index in range(math.ceil((end - start) * fps - 1e-9)) if start + index / fps < end and start + index / fps <= at]
    if not times:
        raise ValueError("No causal frame available")
    if len(times) > max_frames:
        times = [times[index] for index in np.linspace(0, len(times) - 1, max_frames).round().astype(int)]
    return times


def doc_to_messages(doc: dict, lmms_eval_specific_kwargs: dict | None = None, *, round_idx: int | None = None, previous_output: list[str] | None = None, previous_round_info=None):
    """Build independent point queries or an ordered SG/MD session round."""
    index = round_idx or 0
    if index >= len(doc["qa"]):
        return None, True, previous_output, previous_round_info
    qa = doc["qa"][index]
    question = qa["question"]
    if doc["task"] == "md" and index:
        if previous_output is None or len(previous_output) != index:
            raise ValueError("MD requires the preceding actual predictions")
        previous = previous_output[-1]
        question = question.replace("##ANSWER##", "" if failed(previous) else previous)
    profile = doc["profile"]
    times = frame_times(qa, int(profile.get("max_frames", 360)), float(profile.get("target_fps", 1)))
    entry = Path(profile["frame_root"]) / doc["key"]
    _, _, manifest = pack_index(str(entry.resolve()))
    duration = manifest.get("video", {}).get("duration")
    if duration is not None and qa["at_sec"] > duration + 1e-6:
        raise ValueError("Query is outside video duration; source annotation retained")
    frames = read_frames(entry, times)
    messages, content = [], []
    for index_in_video, (timestamp, blob) in enumerate(zip(times, frames, strict=True)):
        content.extend([{"type": "text", "text": f"<{timestamp:.1f} seconds>"}, image_content(blob)])
        if profile["history_layout"] == "per_frame_silence" and index_in_video < len(times) - 1:
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": profile.get("silence_token", "<silence>")}]})
            content = []
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    return messages if round_idx is None else (messages, False, previous_output, previous_round_info)


def doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Provide text context for ordering without reading frames twice."""
    return doc["qa"][0]["question"]


def doc_to_target(doc: dict) -> str:
    """Expose targets only to the evaluation/scoring side."""
    return json.dumps([qa["answer"] for qa in doc["qa"]], ensure_ascii=False)
