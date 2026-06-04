"""OpenAI-compatible gateway for RaaS (Phase 1: single-turn).

External agentic harnesses (Harbor/Terminus-2 today; any OpenAI client) point
their ``api_base`` at RaaS instead of talking to SGLang directly. RaaS tokenizes
the request itself (chat template), drives generation through the existing
``engine.agenerate`` path (native ``/generate`` — token ids in, ids+logprobs
out), and builds the OpenAI response. This keeps every request inside the RaaS
policy layer (concurrency, pause/weight-update coordination, load balancing,
per-token version stamping) instead of bypassing it.

Phase 1 implements **stateless single-turn** chat/completions. The ``/ep/{id}``
path prefix is accepted (so harnesses can be pointed at the per-episode base
URL now) but the per-episode trajectory ledger and multi-turn reconstruction
land in Phase 2; ``episode_id`` is currently only echoed in logs.

See ``claude-doc/OPENAI_GATEWAY_PLAN.md``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from astraflow.raas.api.io_struct import ModelRequest
from astraflow.raas.utils import logging

logger = logging.getLogger(__name__)


def _error(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": None}},
    )


def _finish_reason(stop_reason: str | None) -> str:
    # engine maps backend stops to {"stop","tool_calls","length","interrupt"};
    # abort is already folded into "length" by agenerate.
    return "length" if stop_reason == "length" else "stop"


def _encode_prompt(tokenizer, messages: list[dict], tools: Any, enable_thinking: bool | None) -> list[int]:
    from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids

    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return apply_chat_template_to_ids(tokenizer, messages, **kwargs)


async def _handle_chat(
    manager: Any, body: dict[str, Any], episode_id: str | None
) -> JSONResponse:
    if not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _error(400, "'messages' must be a non-empty list.")

    model = body.get("model")
    try:
        engine, tokenizer, base_gconfig = manager.pick_engine(model)
    except KeyError as exc:
        return _error(404, str(exc), err_type="model_not_found")
    except RuntimeError as exc:
        return _error(503, str(exc), err_type="server_not_ready")

    n = int(body.get("n", 1) or 1)
    if n < 1:
        return _error(400, "'n' must be >= 1.")

    # extra_body fields ride either at top level (LiteLLM flattens) or nested.
    extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
    enable_thinking = body.get("enable_thinking", extra.get("enable_thinking"))

    gconfig = manager.build_gconfig(base_gconfig, tokenizer, body)
    loop = asyncio.get_event_loop()
    ledger = getattr(manager, "_ledger", None)
    use_episode = episode_id is not None and ledger is not None
    if use_episode and n != 1:
        return _error(400, "Episode (/ep/{id}) requests must use n=1.")

    def _make_req(ids) -> ModelRequest:
        return ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=list(ids),
            gconfig=gconfig.new(n_samples=1),
            tokenizer=tokenizer,
        )

    try:
        async with manager.proxy_slot():
            if use_episode:
                # Stateful: ledger builds the exact continuation prompt and
                # records this turn (bit-exact token-in-token-out).
                input_ids = await loop.run_in_executor(
                    None,
                    lambda: ledger.prepare_turn(
                        episode_id, messages, tokenizer,
                        model_id=model, tools=body.get("tools"),
                        enable_thinking=enable_thinking,
                    ),
                )
                if not input_ids:
                    return _error(400, "Chat template produced an empty prompt.")
                async with engine.managed_session():
                    resp = await engine.agenerate(_make_req(input_ids))
                ledger.append_completion(episode_id, resp, tokenizer)
                responses = [resp]
            else:
                # Stateless single-turn.
                input_ids = await loop.run_in_executor(
                    None, _encode_prompt, tokenizer, messages, body.get("tools"), enable_thinking
                )
                if not input_ids:
                    return _error(400, "Chat template produced an empty prompt.")
                async with engine.managed_session():
                    responses = await asyncio.gather(
                        *[engine.agenerate(_make_req(input_ids)) for _ in range(n)]
                    )
    except Exception as exc:  # noqa: BLE001 — surface as an OpenAI-shaped 500
        logger.exception("OpenAI gateway generation failed (episode=%s)", episode_id)
        return _error(500, repr(exc), err_type="internal_error")

    choices = []
    completion_tokens = 0
    prompt_tokens = len(responses[0].input_tokens) if responses else len(input_ids)
    for idx, resp in enumerate(responses):
        text = await loop.run_in_executor(
            None, lambda r=resp: tokenizer.decode(r.output_tokens, skip_special_tokens=True)
        )
        completion_tokens += len(resp.output_tokens)
        choices.append(
            {
                "index": idx,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": _finish_reason(resp.stop_reason),
            }
        )

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or manager.default_model_name(),
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def _models_payload(manager: Any) -> dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": created, "owned_by": "astraflow"}
            for mid in manager.served_model_names()
        ],
    }


def register_openai_routes(app: FastAPI) -> None:
    """Mount the OpenAI-compatible routes onto the RaaS FastAPI app."""

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        return await _handle_chat(request.app.state.manager, body, None)

    @app.post("/ep/{episode_id}/v1/chat/completions")
    async def chat_completions_ep(episode_id: str, request: Request):
        body = await request.json()
        return await _handle_chat(request.app.state.manager, body, episode_id)

    @app.get("/v1/models")
    async def models(request: Request):
        return _models_payload(request.app.state.manager)

    @app.get("/ep/{episode_id}/v1/models")
    async def models_ep(episode_id: str, request: Request):
        return _models_payload(request.app.state.manager)
