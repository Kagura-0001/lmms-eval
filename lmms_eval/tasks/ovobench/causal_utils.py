"""Frozen local OVO point queries; existing upstream OVO tasks are untouched."""

import re
from collections import defaultdict
from statistics import mean

from lmms_eval.tasks.streamingbench_ovobench_omnimmi.utils import failed

TASK_MODES = {"EPM": "backward", "ASI": "backward", "HLD": "backward", "STU": "realtime", "OJR": "realtime", "ATR": "realtime", "ACR": "realtime", "OCR": "realtime", "FPD": "realtime", "REC": "forward", "SSR": "forward", "CRR": "forward"}


def process_results(doc: dict, results: list) -> dict:
    """Preserve source matching, including count concatenation and case."""
    prediction = results[0].strip()
    answer, task = doc["qa"][0]["answer"].strip(), doc["task"]
    error = failed(prediction)
    if task == "REC":
        correct = "".join(re.findall(r"\d+", prediction)) == answer
    else:
        if task in {"SSR", "CRR"}:
            prediction = {"Y": "Yes", "N": "No"}.get(prediction, prediction)
        correct = answer in prediction
    return {"accuracy": {"task": task, "score": float(correct and not error)}, "inference_failures": int(error)}


def aggregate(values: list[dict]) -> float:
    """Average within task, then within mode, then across modes."""
    tasks, modes = defaultdict(list), defaultdict(list)
    for row in values:
        tasks[row["task"]].append(row["score"])
    for task, scores in tasks.items():
        modes[TASK_MODES[task]].append(mean(scores))
    return mean(mean(scores) for scores in modes.values())
