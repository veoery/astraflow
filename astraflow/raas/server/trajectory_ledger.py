"""Server-side per-episode token ledger for the RaaS OpenAI gateway.

Reconstructs **bit-exact, token-in-token-out** multi-turn trajectories from the
sequence of ``/v1/chat/completions`` calls in one episode, without trusting
anything the harness reports. The exact generated token ids are carried forward
across turns; only the genuinely new (tool/user) message delta is tokenized.

Delta tokenization uses the **terminator-anchored** boundary (cf.
``core/workflow/impl/agentbench/task_server.py``), hardened so it does not
assume ``tokenizer.eos_token_id`` is the chat turn terminator (the terminator id
is *derived* from the template). See ``claude-doc/OPENAI_GATEWAY_PLAN.md`` §6 and the
"Delta slice boundary" note.

v1 scope: linear, append-only single-conversation episodes. Non-linear histories
(subagents, edits) are detected and the episode is flagged ``degraded``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from astraflow.raas.utils import logging

logger = logging.getLogger(__name__)


def _render(tokenizer, messages, *, add_generation_prompt, enable_thinking, tools=None):
    from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids

    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return apply_chat_template_to_ids(tokenizer, messages, **kwargs)


def _messages_key(messages: list[dict]) -> list[tuple]:
    """Comparable view of a message list for linear-append checking."""
    out = []
    for m in messages:
        out.append((m.get("role"), m.get("content"), str(m.get("tool_call_id", "")), m.get("name")))
    return out


@dataclass
class EpisodeState:
    model_id: str | None
    enable_thinking: bool | None = None
    seq: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    last_messages_key: list[tuple] = field(default_factory=list)
    last_output_text: str = ""
    pending_input_len: int | None = None  # len(seq) expected as the next prompt
    turns: int = 0
    degraded: bool = False
    created_at: float = field(default_factory=time.monotonic)
    touched_at: float = field(default_factory=time.monotonic)


class TrajectoryLedger:
    """Holds per-episode token state and reconstructs training trajectories."""

    def __init__(self) -> None:
        self._episodes: dict[str, EpisodeState] = {}
        # cache of (terminator_id, content_end_idx) per (id(tokenizer), enable_thinking)
        self._suffix_cache: dict[tuple[int, Any], tuple[int, int]] = {}

    # -- lifecycle ---------------------------------------------------------

    def open(self, episode_id: str, model_id: str | None) -> str:
        self._episodes[episode_id] = EpisodeState(model_id=model_id)
        return episode_id

    def has(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    def ensure(self, episode_id: str, model_id: str | None) -> EpisodeState:
        st = self._episodes.get(episode_id)
        if st is None:
            st = EpisodeState(model_id=model_id)
            self._episodes[episode_id] = st
        return st

    def close(self, episode_id: str) -> None:
        self._episodes.pop(episode_id, None)

    def gc(self, ttl_s: float = 3600.0, max_episodes: int = 4096) -> int:
        now = time.monotonic()
        dropped = 0
        for eid in list(self._episodes):
            if now - self._episodes[eid].touched_at > ttl_s:
                self._episodes.pop(eid, None)
                dropped += 1
        if len(self._episodes) > max_episodes:
            for eid in sorted(self._episodes, key=lambda e: self._episodes[e].touched_at)[
                : len(self._episodes) - max_episodes
            ]:
                self._episodes.pop(eid, None)
                dropped += 1
        return dropped

    # -- template terminator derivation -----------------------------------

    def _terminator(self, tokenizer, enable_thinking) -> tuple[int, int]:
        """Return ``(terminator_id, content_end_idx)`` for an assistant turn.

        ``terminator_id`` is the token the template appends right after assistant
        *content* (e.g. ``<|im_end|>``), derived by diffing two assistant renders
        so it does not assume ``eos_token_id``. ``content_end_idx`` is the index
        of that terminator within ``render([{assistant: "A"}])``.
        """
        key = (id(tokenizer), enable_thinking)
        cached = self._suffix_cache.get(key)
        if cached is not None:
            return cached
        a = _render(tokenizer, [{"role": "assistant", "content": "A"}],
                    add_generation_prompt=False, enable_thinking=enable_thinking)
        b = _render(tokenizer, [{"role": "assistant", "content": "ZZZZ"}],
                    add_generation_prompt=False, enable_thinking=enable_thinking)
        # longest common suffix of the two renders = the assistant closing block
        i = 0
        while i < min(len(a), len(b)) and a[-1 - i] == b[-1 - i]:
            i += 1
        if i == 0:
            raise ValueError("Could not derive assistant turn terminator from chat template.")
        content_end_idx = len(a) - i           # index where the closing block starts in `a`
        terminator_id = a[content_end_idx]      # first token of the closing block
        result = (terminator_id, content_end_idx)
        self._suffix_cache[key] = result
        return result

    def _continuation_delta(self, tokenizer, delta_msgs, enable_thinking) -> tuple[int, list[int]]:
        """Tokens to append AFTER the assistant terminator for ``delta_msgs``.

        Returns ``(terminator_id, delta_after_terminator)`` where the delta is
        ``[\\n, <rendered delta_msgs>, <next generation prompt>]`` — i.e. the
        ``task_server.py`` slice, computed for these specific messages.
        """
        terminator_id, content_end_idx = self._terminator(tokenizer, enable_thinking)
        full = _render(
            tokenizer,
            [{"role": "assistant", "content": "A"}] + list(delta_msgs),
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if full[content_end_idx] != terminator_id:
            # Template rendered the anchor differently in context (e.g. variable
            # system prepend) — caller will mark the episode degraded.
            raise ValueError("Anchor/full prefix mismatch while computing turn delta.")
        return terminator_id, full[content_end_idx + 1:]

    # -- per-turn API (called by the gateway) ------------------------------

    def prepare_turn(
        self,
        episode_id: str,
        messages: list[dict],
        tokenizer,
        *,
        model_id: str | None = None,
        tools: Any = None,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        """Build the exact ``input_ids`` for this turn and record the prompt span."""
        st = self.ensure(episode_id, model_id)
        st.touched_at = time.monotonic()
        mk = _messages_key(messages)

        if not st.seq:
            st.enable_thinking = enable_thinking
            input_ids = _render(
                tokenizer, messages, add_generation_prompt=True,
                enable_thinking=enable_thinking, tools=tools,
            )
            st.seq = list(input_ids)
            st.loss_mask = [0] * len(input_ids)
            st.logprobs = [0.0] * len(input_ids)
            st.versions = [-1] * len(input_ids)
            st.last_messages_key = mk
            st.pending_input_len = len(st.seq)
            return list(st.seq)

        # Continuation: require linear append.
        prev = st.last_messages_key
        if mk[: len(prev)] != prev:
            logger.warning(
                "Episode %s: non-linear message history (len prev=%d, new=%d); "
                "marking degraded and re-encoding from scratch.",
                episode_id, len(prev), len(mk),
            )
            st.degraded = True
            input_ids = _render(
                tokenizer, messages, add_generation_prompt=True,
                enable_thinking=st.enable_thinking, tools=tools,
            )
            st.seq = list(input_ids)
            st.loss_mask = [0] * len(input_ids)
            st.logprobs = [0.0] * len(input_ids)
            st.versions = [-1] * len(input_ids)
            st.last_messages_key = mk
            st.pending_input_len = len(st.seq)
            return list(st.seq)

        new_msgs = messages[len(prev):]
        # Drop the assistant turn we just generated (we already hold its exact ids).
        if new_msgs and new_msgs[0].get("role") == "assistant":
            delta_msgs = new_msgs[1:]
        else:
            delta_msgs = new_msgs

        try:
            terminator_id, delta_after = self._continuation_delta(
                tokenizer, delta_msgs, st.enable_thinking
            )
        except ValueError:
            logger.warning("Episode %s: delta computation failed; degrading.", episode_id)
            st.degraded = True
            input_ids = _render(
                tokenizer, messages, add_generation_prompt=True,
                enable_thinking=st.enable_thinking, tools=tools,
            )
            st.seq = list(input_ids)
            st.loss_mask = [0] * len(input_ids)
            st.logprobs = [0.0] * len(input_ids)
            st.versions = [-1] * len(input_ids)
            st.last_messages_key = mk
            st.pending_input_len = len(st.seq)
            return list(st.seq)

        # Normalize the terminator exactly once: if the model already emitted it
        # (kept mask=1, trainable), don't re-add; otherwise add it as mask=0.
        new_tail: list[int] = []
        if not st.seq or st.seq[-1] != terminator_id:
            new_tail.append(terminator_id)
        new_tail.extend(delta_after)

        st.seq.extend(new_tail)
        st.loss_mask.extend([0] * len(new_tail))
        st.logprobs.extend([0.0] * len(new_tail))
        st.versions.extend([-1] * len(new_tail))
        st.last_messages_key = mk
        st.pending_input_len = len(st.seq)
        return list(st.seq)

    def append_completion(self, episode_id: str, resp, tokenizer) -> None:
        """Record the generated tokens (mask=1) after ``prepare_turn``."""
        st = self._episodes.get(episode_id)
        if st is None:
            return
        st.touched_at = time.monotonic()
        # Sanity: what the engine conditioned on must equal what we built.
        if st.pending_input_len is not None and len(resp.input_tokens) != st.pending_input_len:
            logger.warning(
                "Episode %s: prompt length drift (built=%s, engine saw=%s); degrading.",
                episode_id, st.pending_input_len, len(resp.input_tokens),
            )
            st.degraded = True
        st.seq.extend(list(resp.output_tokens))
        st.loss_mask.extend([1] * len(resp.output_tokens))
        st.logprobs.extend(list(resp.output_logprobs))
        st.versions.extend(list(resp.output_versions))
        st.last_output_text = tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
        st.turns += 1
        st.pending_input_len = None

    # -- trajectory readout (called by the workflow) -----------------------

    def get_trajectory(self, episode_id: str) -> dict[str, Any] | None:
        """Return the reconstructed trajectory as plain lists (or ``None``)."""
        st = self._episodes.get(episode_id)
        if st is None or not st.seq:
            return None
        assert len(st.seq) == len(st.loss_mask) == len(st.logprobs) == len(st.versions)
        return {
            "input_ids": list(st.seq),
            "loss_mask": list(st.loss_mask),
            "logprobs": list(st.logprobs),
            "versions": list(st.versions),
            "turns": st.turns,
            "degraded": st.degraded,
        }
