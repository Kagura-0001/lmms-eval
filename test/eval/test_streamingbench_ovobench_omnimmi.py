"""Protocol tests that catch future-frame, gold-answer and denominator errors."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lmms_eval.tasks.omnimmi import utils as omni
from lmms_eval.tasks.ovobench.causal_utils import aggregate
from lmms_eval.tasks.streamingbench_ovobench_omnimmi import utils


def test_half_open_and_uniform_grid():
    assert utils.frame_times({"range_sec": [0, 3], "at_sec": 3}, 360) == [0, 1, 2]
    assert utils.frame_times({"range_sec": [0, 4], "at_sec": 3}, 360) == [0, 1, 2, 3]
    assert utils.frame_times({"range_sec": [0, 1000], "at_sec": 999}, 3) == [0, 500, 999]


def test_md_prediction_and_layout(monkeypatch):
    monkeypatch.setattr(utils, "pack_index", lambda path: (None, None, {"video": {"duration": 100}}))
    monkeypatch.setattr(utils, "read_frames", lambda entry, times: [b"jpeg"] * len(times))
    doc = {
        "task": "md",
        "key": "key",
        "profile": {"frame_root": "/unused", "history_layout": "packed"},
        "qa": [{"question": "First?", "answer": "SECRET_GT", "range_sec": [0, 2], "at_sec": 2}, {"question": "After ##ANSWER##?", "answer": "SECRET_GT2", "range_sec": [0, 2], "at_sec": 2}],
    }
    packed = utils.doc_to_messages(doc, round_idx=1, previous_output=["actual prediction"])[0]
    assert packed[-1]["content"][-1]["text"] == "After actual prediction?"
    assert "SECRET_GT" not in str(packed)
    variant = deepcopy(doc)
    variant["profile"]["history_layout"] = "per_frame_silence"
    split = utils.doc_to_messages(variant, round_idx=1, previous_output=["actual prediction"])[0]
    assert split[1]["content"][0]["text"] == "<silence>"
    assert [part for msg in split if msg["role"] == "user" for part in msg["content"]] == packed[0]["content"]
    assert utils.doc_to_messages(doc, round_idx=2, previous_output=["a", "b"])[1]
    with pytest.raises(ValueError, match="preceding"):
        utils.doc_to_messages(doc, round_idx=1)


def test_out_of_duration_is_not_clamped(monkeypatch):
    monkeypatch.setattr(utils, "pack_index", lambda path: (None, None, {"video": {"duration": 1}}))
    doc = {"task": "si", "key": "key", "profile": {"frame_root": "/unused", "history_layout": "packed"}, "qa": [{"question": "Who?", "range_sec": [0, 2], "at_sec": 2}]}
    with pytest.raises(ValueError, match="outside video"):
        utils.doc_to_messages(doc)


def test_session_and_position_denominators():
    def record(scores):
        return {"items": [{"correct": score, "score": score * 5, "judge_error": None} for score in scores]}

    values = [record([1, 0]), record([1, 1, 1])]
    assert omni.accuracy(values) == 0.5
    assert omni.item_accuracy_2(values) == 0.5
    assert omni.accuracy_3(values) == 1
    assert omni.score_avg(values) == 3.75  # session mean, not question mean
    values[0]["items"][0]["judge_error"] = "unavailable"
    assert str(omni.accuracy(values)) == "nan"


def test_ovo_macro_averaging():
    values = [{"task": "EPM", "score": 1}] * 9 + [{"task": "ASI", "score": 0}, {"task": "STU", "score": 1}]
    assert aggregate(values) == 0.75


def test_judge_rejects_malformed_output():
    assert omni.parse_judgment("{'pred': 'yes', 'score': 4.8}")["score"] == 4.8
    with pytest.raises(ValueError):
        omni.parse_judgment("{'pred': 'yes', 'score': 20}")


def test_layout_changes_native_task_fingerprint(tmp_path, monkeypatch):
    annotation = b'{"key":"video","qa":[]}\n'
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations/task.jsonl").write_bytes(annotation)
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": {"task": {"sha256": hashlib.sha256(annotation).hexdigest(), "count": 1}}}))
    monkeypatch.setenv("LMMS_STREAMING_DATA_ROOT", str(tmp_path))
    configs = []
    for layout in ("packed", "per_frame_silence"):
        monkeypatch.setenv("LMMS_STREAMING_HISTORY_LAYOUT", layout)
        task = SimpleNamespace(config=SimpleNamespace(task="task", test_split="test", lmms_eval_specific_kwargs={"default": {"history_layout": "packed"}}))
        utils.StreamingBenchmarkTask.download(task)
        configs.append(task.config.lmms_eval_specific_kwargs)
        assert task.dataset["test"][0]["profile"]["history_layout"] == layout
    assert configs[0] != configs[1]
