"""Phase 2 tests — per-episode token ledger + bit-exact multi-turn.

Uses a controllable ChatML-like fake tokenizer as an oracle, plus a focused
check against the real Qwen3-8B tokenizer for the canonical-seam property.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest

from astraflow.raas.server.trajectory_ledger import TrajectoryLedger

# token ids for the fake template
IM_START, IM_END, NL = 1, 2, 3
ROLE = {"system": 4, "user": 5, "assistant": 6, "tool": 7}


class FakeChatTokenizer:
    """Deterministic ChatML-ish template: <im_start> role \\n {chars} <im_end> \\n."""

    eos_token_id = IM_END
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        ids = []
        for m in messages:
            ids += [IM_START, ROLE.get(m["role"], 5), NL]
            ids += [ord(c) for c in m.get("content", "")]
            ids += [IM_END, NL]
        if add_generation_prompt:
            ids += [IM_START, ROLE["assistant"], NL]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if 32 <= i < 0x110000)


@dataclass
class FakeResp:
    input_tokens: list
    output_tokens: list
    output_logprobs: list
    output_versions: list
    stop_reason: str = "stop"


def _gen(ledger, eid, tok, messages, content_tokens, version):
    """Drive one turn: prepare → fake-generate → append. Returns input_ids."""
    input_ids = ledger.prepare_turn(eid, messages, tok, enable_thinking=False)
    resp = FakeResp(
        input_tokens=list(input_ids),
        output_tokens=list(content_tokens),
        output_logprobs=[-1.0] * len(content_tokens),
        output_versions=[version] * len(content_tokens),
    )
    ledger.append_completion(eid, resp, tok)
    return input_ids


def test_bitexact_multiturn_reconstruction_and_mask():
    tok = FakeChatTokenizer()
    led = TrajectoryLedger()
    eid = led.open("e1", "m")

    msgs1 = [{"role": "user", "content": "hi"}]
    in1 = _gen(led, eid, tok, msgs1, [ord("A"), ord("1"), IM_END], version=5)
    assert in1 == tok.apply_chat_template(msgs1, add_generation_prompt=True)

    # Turn 2: harness echoes the assistant text and adds a user message.
    msgs2 = msgs1 + [
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "q2"},
    ]
    in2 = led.prepare_turn(eid, msgs2, tok, enable_thinking=False)

    # BIT-EXACT: the continuation prompt equals the canonical full render —
    # the assistant turn is carried as exact ids and the seam keeps its "\n".
    canonical2 = tok.apply_chat_template(msgs2, add_generation_prompt=True)
    assert in2 == canonical2
    assert IM_END in in2 and in2[in2.index(IM_END) + 1] == NL  # <im_end>\n seam

    resp2 = FakeResp(in2, [ord("B"), IM_END], [-2.0, -2.0], [6, 6])
    led.append_completion(eid, resp2, tok)

    traj = led.get_trajectory(eid)
    assert traj["degraded"] is False
    assert traj["turns"] == 2
    n = len(traj["input_ids"])
    assert n == len(traj["loss_mask"]) == len(traj["logprobs"]) == len(traj["versions"])

    # Every completion token (both turns) is trainable; prompts are masked.
    ones = [i for i, m in enumerate(traj["loss_mask"]) if m == 1]
    assert [traj["input_ids"][i] for i in ones] == [ord("A"), ord("1"), IM_END, ord("B"), IM_END]
    assert [traj["versions"][i] for i in ones] == [5, 5, 5, 6, 6]
    assert all(traj["versions"][i] == -1 for i in range(n) if traj["loss_mask"][i] == 0)


def test_terminator_not_doubled_when_model_emits_eos():
    tok = FakeChatTokenizer()
    led = TrajectoryLedger()
    eid = led.open("e2", "m")
    led_in1 = _gen(led, eid, tok, [{"role": "user", "content": "x"}], [ord("Y"), IM_END], 1)
    msgs2 = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "Y"},
        {"role": "user", "content": "z"},
    ]
    in2 = led.prepare_turn(eid, msgs2, tok, enable_thinking=False)
    # exactly one IM_END between the assistant content and the next message
    assert in2.count(IM_END) == msgs2.__len__()  # one per closed message (user,assistant,user)
    assert in2 == tok.apply_chat_template(msgs2, add_generation_prompt=True)


def test_degraded_on_nonlinear_history():
    tok = FakeChatTokenizer()
    led = TrajectoryLedger()
    eid = led.open("e3", "m")
    _gen(led, eid, tok, [{"role": "user", "content": "hi"}], [ord("A"), IM_END], 1)
    # A non-extending history (different first message) → degraded, still usable.
    bad = [{"role": "user", "content": "totally different"}]
    led.prepare_turn(eid, bad, tok, enable_thinking=False)
    assert led.get_trajectory(eid)["degraded"] is True


def test_terminator_anchored_delta_keeps_newline_qwen():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qwen3-8B tokenizer unavailable: {exc}")
    led = TrajectoryLedger()
    term, _ = led._terminator(tok, False)
    assert term == tok.convert_tokens_to_ids("<|im_end|>")
    term2, delta = led._continuation_delta(tok, [{"role": "user", "content": "obs"}], False)
    assert term2 == term
    assert tok.decode([delta[0]]) == "\n"          # canonical seam preserved
    assert "<|im_start|>user" in tok.decode(delta)


# --- gateway episode path (stateful) ----------------------------------------

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
        from astraflow.raas.api.io_struct import ModelResponse

        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=[ord("Z"), IM_END],
            output_logprobs=[-0.5, -0.5],
            output_versions=[9, 9],
            stop_reason="stop",
        )


def _ready_manager():
    from astraflow.raas.server.manager import RaaS3Manager

    mgr = RaaS3Manager()
    mgr._status = "ready"
    mgr._tokenizer = FakeChatTokenizer()
    mgr._gconfig = FakeGConfig()
    mgr._engine = FakeEngine()
    return mgr


def test_gateway_episode_records_trajectory():
    import asyncio
    import json

    from astraflow.raas.server.openai_gateway import _handle_chat

    mgr = _ready_manager()
    eid, _ = mgr.open_episode("m")
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    resp = asyncio.run(_handle_chat(mgr, body, eid))
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["choices"][0]["message"]["role"] == "assistant"

    traj = mgr.get_trajectory(eid)
    assert traj is not None and traj["turns"] == 1
    ones = [i for i, m in enumerate(traj["loss_mask"]) if m == 1]
    assert [traj["input_ids"][i] for i in ones] == [ord("Z"), IM_END]
    assert [traj["versions"][i] for i in ones] == [9, 9]


def test_gateway_episode_rejects_n_gt_1():
    import asyncio

    from astraflow.raas.server.openai_gateway import _handle_chat

    mgr = _ready_manager()
    eid, _ = mgr.open_episode("m")
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 2}
    resp = asyncio.run(_handle_chat(mgr, body, eid))
    assert resp.status_code == 400


# --- Phase 3: RL workflow sources trajectories from the ledger --------------

def _make_rl_workflow():
    from astraflow.core.workflow.impl import terminal_bench_harbor as tbh

    return tbh, tbh.TerminalBenchHarborRLWorkflow(
        gconfig=FakeGConfig(), tokenizer=None, jobs_dir="/tmp/harbor-test"
    )


def test_rl_workflow_builds_tensors_from_ledger(monkeypatch, tmp_path):
    import asyncio
    from pathlib import Path

    from astraflow.raas.api.io_struct import ModelResponse
    from astraflow.raas.server.manager import RolloutContext, current_rollout

    tbh, wf = _make_rl_workflow()
    mgr = _ready_manager()

    async def fake_trial(data, api_base_override=None):
        # Simulate Harbor hitting the gateway: one turn into THIS episode.
        ep_id = api_base_override.split("/ep/")[1].split("/")[0]
        ids = mgr._ledger.prepare_turn(
            ep_id, [{"role": "user", "content": "hi"}], mgr._tokenizer, enable_thinking=False
        )
        resp = ModelResponse(
            input_tokens=list(ids), output_tokens=[ord("A"), IM_END],
            output_logprobs=[-1.0, -1.0], output_versions=[4, 4], stop_reason="stop",
        )
        mgr._ledger.append_completion(ep_id, resp, mgr._tokenizer)
        return {"jobs_dir": str(tmp_path), "reward": 1.0}

    monkeypatch.setattr(wf, "_run_one_harbor_trial", fake_trial)
    monkeypatch.setattr(
        tbh, "_load_harbor_trial_result",
        lambda job_root: (Path("result.json"),
                          {"agent_result": {}, "verifier_result": {"rewards": {"reward": 1.0}}}),
    )

    token = current_rollout.set(RolloutContext(mgr))
    try:
        out = asyncio.run(wf._run_one_harbor_training_trial({"task_name": "t"}))
    finally:
        current_rollout.reset(token)

    ids = out["input_ids"][0].tolist()
    mask = out["loss_mask"][0].tolist()
    versions = out["versions"][0].tolist()
    trained = [ids[i] for i in range(len(ids)) if mask[i] == 1]
    assert trained == [ord("A"), IM_END]
    assert [versions[i] for i in range(len(ids)) if mask[i] == 1] == [4, 4]
    assert float(out["rewards"][0].item()) == 1.0
    # episode closed in finally → no leak
    assert mgr._ledger._episodes == {}


def test_rl_workflow_requires_rollout_ctx():
    import asyncio

    _tbh, wf = _make_rl_workflow()
    with pytest.raises(RuntimeError, match="requires the RaaS OpenAI gateway"):
        asyncio.run(wf._run_one_harbor_training_trial({}))
