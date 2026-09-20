"""Frozen StreamingBench point-query parsing, shared by all five tasks."""

import re

from lmms_eval.tasks.streamingbench_ovobench_omnimmi.utils import failed


def mcq(value: str) -> str:
    """Preserve the local source parser, including its fallback letter rule."""
    text = value.strip().upper()
    if re.fullmatch(r"[A-Z]", text):
        return text
    patterns = (r"ANSWER\s+IS[:\s]+([A-Z])\b", r"(?:CHOOSE|SELECT|PICK)\s+(?:OPTION\s+)?([A-Z])\b", r"(?:^|\b)([A-Z])(?:\s*[.)。：:,，]|\s|$)")
    return next((match.group(1) for pattern in patterns if (match := re.search(pattern, text))), "")


def process_results(doc: dict, results: list) -> dict:
    """Keep failed candidates in the case-accuracy denominator."""
    prediction = results[0]
    parsed, expected = mcq(prediction), mcq(doc["qa"][0]["answer"])
    error = failed(prediction)
    return {"accuracy": float(not error and bool(parsed and expected and parsed == expected)), "inference_failures": int(error)}


def failure_count(values: list[int]) -> int:
    """Count failures rather than silently filtering them."""
    return sum(values)
