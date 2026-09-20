"""Exercise generic ordered rounds with concurrent independent sessions."""

import asyncio
from types import SimpleNamespace

from lmms_eval.api.instance import Instance, TokenCounts
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat


def test_sessions_concurrent_rounds_ordered():
    backend = object.__new__(AsyncOpenAIChat)
    backend.num_cpus = 2
    backend.api_key = "EMPTY"
    backend.base_url = "http://localhost:1/v1"
    backend.timeout = 1
    backend.max_retries = 1
    backend.retry_backoff_s = 0
    backend.task_dict = {"task": {"test": [{"id": 0}, {"id": 1}]}}
    active = 0
    peak = 0
    calls = []

    def callback(doc, round_idx=None, previous_output=None, previous_round_info=None):
        if round_idx == 2:
            return None, True, previous_output, previous_round_info
        if round_idx is None:
            return [str(doc["id"])]
        assert previous_output == [f"answer-{doc['id']}"]
        return [previous_output[-1]], False, previous_output, None

    async def forward(request, index):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append((index, request.args[1](None)))
        await asyncio.sleep(0.01)
        active -= 1
        return f"answer-{index}", index, TokenCounts()

    backend.maybe_forward_with_tool = forward
    requests = [Instance("generate_until_multi_round", ("", callback, {}, index, "task", "test"), 0, {"task": "task", "doc_id": index, "repeats": 1}) for index in range(2)]
    assert backend.generate_until_multi_round(requests) == [["answer-0", "answer-0"], ["answer-1", "answer-1"]]
    assert peak == 2
    assert [content for index, content in calls if index == 0] == [["0"], ["answer-0"]]


def test_judge_provider_forwards_extra_body(monkeypatch):
    from lmms_eval.llm_judge.protocol import Request, ServerConfig
    from lmms_eval.llm_judge.providers.openai import OpenAIProvider

    provider = object.__new__(OpenAIProvider)
    provider.api_key = "EMPTY"
    provider.config = ServerConfig(model_name="judge", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    provider.use_client = True
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))], model="judge", usage=None)

    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert provider.evaluate(Request(messages=[])).success
    assert captured["extra_body"] == provider.config.extra_body


def test_connections_closed_before_each_event_loop_exits(monkeypatch):
    clients = []

    class Client:
        def __init__(self, **kwargs):
            self.loop = asyncio.get_running_loop()
            self.closed = False
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            assert self.loop is asyncio.get_running_loop()
            self.closed = True

    monkeypatch.setattr("lmms_eval.models.chat.async_openai.AsyncOpenAI", Client)
    backend = object.__new__(AsyncOpenAIChat)
    backend.api_key, backend.base_url, backend.timeout = "EMPTY", None, 1

    async def run():
        assert backend.client.loop is asyncio.get_running_loop()
        assert not backend.client.closed
        return "answer"

    for _ in range(2):
        assert asyncio.run(backend._run_with_client(run)) == "answer"
        assert clients[-1].closed
        assert clients[-1].loop.is_closed()
    assert clients[0].loop is not clients[1].loop


def test_forward_does_not_mutate_cache_inputs():
    backend = object.__new__(AsyncOpenAIChat)
    backend.task_dict = {"task": {"test": [{}]}}
    backend.max_pixels, backend.min_pixels, backend.max_frames = 100, 10, 1
    backend.fps, backend.nframes = 1, None
    backend.is_qwen3_vl, backend.system_prompt, backend.mcp_client = False, None, None
    backend.model_version = "test"

    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")], usage=None)

    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    kwargs = {"temperature": 0}
    request = Instance("generate_until", ("", lambda doc: [{"role": "user", "content": [{"type": "text", "text": "question"}]}], kwargs, 0, "task", "test"), 0, {"task": "task", "doc_id": 0, "repeats": 1})
    assert asyncio.run(backend.maybe_forward_with_tool(request, 0))[0] == "answer"
    assert kwargs == {"temperature": 0}
