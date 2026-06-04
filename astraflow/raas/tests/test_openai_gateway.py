"""Phase 1 gateway tests — stateless single-turn /v1/chat/completions.

Uses fake tokenizer/engine so no real SGLang is required.
"""

import asyncio
import json
from contextlib import asynccontextmanager

from astraflow.raas.api.io_struct import ModelResponse
from astraflow.raas.server.manager import RaaS3Manager
from astraflow.raas.server.openai_gateway import _handle_chat, _models_payload


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 98

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        ids = []
        for m in messages:
            ids.append(ord(m["role"][0]))
            ids.append(len(m.get("content", "")))
        if add_generation_prompt:
            ids.append(1)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "decoded:" + ",".join(str(i) for i in ids)


class FakeGConfig:
    n_samples = 1

    def new_with_stop_and_pad_token_ids(self, tokenizer):
        return self

    def new(self, **kwargs):
        return self


class FakeEngine:
    @asynccontextmanager
    async def managed_session(self):
        yield

    async def agenerate(self, req):
        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=[7, 8, 9],
            output_logprobs=[-0.1, -0.2, -0.3],
            output_versions=[3, 3, 3],
            stop_reason="stop",
        )


def _ready_manager():
    mgr = RaaS3Manager()
    mgr._status = "ready"
    mgr._tokenizer = FakeTokenizer()
    mgr._gconfig = FakeGConfig()
    mgr._engine = FakeEngine()  # property setter stores into _engines["default"]
    return mgr


def test_single_turn_chat_completion_envelope():
    mgr = _ready_manager()
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 16,
        "n": 2,
    }
    # episode_id=None → stateless path (n>1 multi-choice is only valid here;
    # episode/stateful requests are restricted to n=1).
    resp = asyncio.run(_handle_chat(mgr, body, None))
    assert resp.status_code == 200
    data = json.loads(resp.body)

    assert data["object"] == "chat.completion"
    assert data["model"] == "m"
    assert len(data["choices"]) == 2
    for ch in data["choices"]:
        assert ch["message"]["role"] == "assistant"
        assert ch["message"]["content"].startswith("decoded:")
        assert ch["finish_reason"] == "stop"
    # 3 output tokens * 2 choices; prompt tokens = encoded prompt length
    assert data["usage"]["completion_tokens"] == 6
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


def test_missing_messages_returns_400():
    mgr = _ready_manager()
    resp = asyncio.run(_handle_chat(mgr, {"model": "m"}, None))
    assert resp.status_code == 400


def test_not_ready_returns_503():
    mgr = _ready_manager()
    mgr._status = "starting"
    resp = asyncio.run(
        _handle_chat(mgr, {"messages": [{"role": "user", "content": "x"}]}, None)
    )
    assert resp.status_code == 503


def test_build_gconfig_maps_openai_params():
    from astraflow.raas.api.cli_args import GenerationHyperparameters

    g = RaaS3Manager.build_gconfig(
        GenerationHyperparameters(),
        FakeTokenizer(),
        {"temperature": 0.0, "max_tokens": 32, "top_p": 0.9, "stop": "END"},
    )
    assert g.n_samples == 1
    assert g.greedy is True            # temperature == 0 → greedy
    assert g.max_new_tokens == 32
    assert g.top_p == 0.9
    assert g.stop == ["END"]


def test_models_payload_lists_served_models():
    mgr = _ready_manager()
    payload = _models_payload(mgr)
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "default"
    assert payload["data"][0]["object"] == "model"
