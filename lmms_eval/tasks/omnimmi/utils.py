"""Original OmniMMI semantic prompts and per-session/cumulative metrics."""

import ast
import math
import os
import re
from functools import lru_cache
from statistics import mean

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.llm_judge.protocol import Request
from lmms_eval.tasks.streamingbench_ovobench_omnimmi.utils import failed

SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs. Your task is to compare the "
    "predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:------##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct "
    "answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate "
    "the correctness of the prediction compared to the answer."
)


def judge_messages(question: str, answer: str, prediction: str) -> list[dict]:
    """Use the frozen source question, including MD's unsubstituted marker."""
    prompt = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\nCorrect Answer: {answer}\nPredicted Answer: {prediction}\n\n"
        "Provide your evaluation only as a yes/no and score where the score "
        "is an integer value between 0 and 5, with 5 indicating the highest "
        "meaningful match. Please generate the response in the form of a "
        "Python dictionary string with keys 'pred' and 'score', where value "
        "of 'pred' is a string of 'yes' or 'no' and value of 'score' is in "
        "INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR "
        "EXPLANATION. Only provide the Python dictionary string without "
        "codeblock. For example, your response should look like this: "
        "{'pred': 'yes', 'score': 4.8}."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def parse_judgment(text: str) -> dict:
    """Validate the native yes/no and 0..5 dictionary response."""
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    value = ast.literal_eval(match.group(0)) if match else None
    if not isinstance(value, dict) or set(value) != {"pred", "score"}:
        raise ValueError("Judge must return pred/score dictionary")
    pred, score = str(value["pred"]).strip().lower(), value["score"]
    if pred not in {"yes", "no"} or isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 5:
        raise ValueError("Invalid judgment")
    return {"pred": pred, "score": float(score)}


@lru_cache(maxsize=1)
def judge():
    """Use lmms-eval's existing provider and independently configured endpoint."""
    if not os.environ.get("OPENAI_API_URL"):
        raise ValueError("Set OPENAI_API_URL to the Qwen3.5-9B judge endpoint")
    return get_server(server_name="openai", config=ServerConfig(model_name="Qwen/Qwen3.5-9B", max_tokens=128, temperature=0, num_retries=3, retry_delay=1, timeout=300, extra_body={"chat_template_kwargs": {"enable_thinking": False}}))


def process_results(doc: dict, results: list) -> dict:
    """Retain individual judgments in sample logs and source denominators."""
    predictions = results[0] if isinstance(results[0], list) else [results[0]]
    if len(predictions) != len(doc["qa"]):
        raise ValueError("Prediction count does not match the complete session")
    items = []
    for qa, prediction in zip(doc["qa"], predictions, strict=True):
        if failed(prediction):
            items.append({"correct": 0, "score": 0 if doc["task"] in {"sg", "md", "si"} else None, "failure": 1, "judge_error": None})
            continue
        try:
            response = judge().evaluate(Request(messages=judge_messages(qa["problem"], qa["answer"], prediction)))
            if not response.success:
                raise ValueError(response.error_message)
            parsed = parse_judgment(response.content)
            items.append({"correct": int(parsed["pred"] == "yes"), "score": parsed["score"], "failure": 0, "judge_error": None, "judge_response": response.content, "judge_model": response.model_used})
        except Exception as exc:
            items.append({"correct": None, "score": None, "failure": 0, "judge_error": f"{type(exc).__name__}: {exc}"})
    record = {"uid": doc["uid"], "items": items}
    output = {"accuracy": record, "score_avg": record, "inference_failures": sum(item["failure"] for item in items), "judge_failures": sum(item["judge_error"] is not None for item in items)}
    if doc["task"] in {"sg", "md"}:
        for index in range(1, 5 if doc["task"] == "sg" else 4):
            output[f"item_accuracy_{index}"] = record
            output[f"cumulative_item_accuracy_{index}"] = record
    return output


def _pending(values: list[dict]) -> bool:
    return any(item["judge_error"] is not None for value in values for item in value["items"])


def accuracy(values: list[dict]) -> float:
    """Whole-session correctness; any pending judge prevents a final score."""
    return float("nan") if _pending(values) else mean(all(item["correct"] for item in value["items"]) for value in values)


def score_avg(values: list[dict]) -> float:
    """Integer-truncate ratings, average within session, then across sessions."""
    scores = [mean(int(item["score"]) for item in value["items"] if item["score"] is not None) for value in values if any(item["score"] is not None for item in value["items"])]
    return float("nan") if _pending(values) or not scores else mean(scores)


def round_accuracy(values: list[dict], index: int, cumulative: bool = True) -> float:
    """Cumulative accuracy among sessions containing this round."""
    selected = [value for value in values if len(value["items"]) >= index]
    if not selected or _pending(selected):
        return float("nan")
    return mean(all(item["correct"] for item in value["items"][:index]) if cumulative else value["items"][index - 1]["correct"] for value in selected)


def accuracy_1(values: list[dict]) -> float:
    """Native cumulative first-round accuracy."""
    return round_accuracy(values, 1)


def accuracy_2(values: list[dict]) -> float:
    """Native cumulative second-round accuracy."""
    return round_accuracy(values, 2)


def accuracy_3(values: list[dict]) -> float:
    """Native cumulative third-round accuracy."""
    return round_accuracy(values, 3)


def accuracy_4(values: list[dict]) -> float:
    """Native cumulative fourth-round accuracy (SG only)."""
    return round_accuracy(values, 4)


def item_accuracy_1(values: list[dict]) -> float:
    """Native per-question accuracy at position 1."""
    return round_accuracy(values, 1, cumulative=False)


def item_accuracy_2(values: list[dict]) -> float:
    """Native per-question accuracy at position 2."""
    return round_accuracy(values, 2, cumulative=False)


def item_accuracy_3(values: list[dict]) -> float:
    """Native per-question accuracy at position 3."""
    return round_accuracy(values, 3, cumulative=False)


def item_accuracy_4(values: list[dict]) -> float:
    """Native per-question accuracy at position 4."""
    return round_accuracy(values, 4, cumulative=False)
