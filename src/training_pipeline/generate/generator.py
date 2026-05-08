"""Drive an LLM through a propose/observe loop and emit canonical trajectories.

Given a seed query, ``TrajectoryGenerator`` runs an open LLM through a
classic agent loop: the model proposes a tool call, the registry returns a
mock observation, the model consumes the result and either continues or
emits a final answer. Every step records the prompt, raw response, parsed
tool call, observation, latency, and finish reason so post-hoc analysis
can attribute regressions to specific steps.

Three LLM backends are supported:

- ``StubLLMBackend`` — deterministic, dependency-free. Picks tools by simple
  keyword matching on the seed query. Useful for tests and CI; useful as a
  smoke-test of the rest of the pipeline.
- ``TransformersLLMBackend`` — lazy ``transformers.AutoModelForCausalLM``
  for local development on a single GPU.
- ``VLLMBackend`` — lazy ``vllm.LLM`` for production batched generation
  on the 8xH100 box specified in the proposal.

A trajectory is *dropped* (returns ``None``) if the model emits arguments
that fail the tool-schema check. The mentor's data-quality bar means we
prefer fewer trajectories of higher fidelity over many noisy ones.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from training_pipeline.generate.mock_tools import MockToolRegistry
from training_pipeline.generate.seeds import Seed
from training_pipeline.ingest.parsers import iter_records, write_jsonl
from training_pipeline.schemas.events import (
    AssistantEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Trajectory,
    UserEvent,
)
from training_pipeline.validate.consistency import validate_tool_call

log = logging.getLogger(__name__)


@dataclass
class GenerationStep:
    """Per-step diagnostic record. Persisted on the trajectory metadata so
    post-hoc analysis can attribute pipeline regressions to specific steps."""

    step: int
    prompt: str
    raw_response: str
    parsed_tool_call: dict[str, Any] | None
    observation: str | None
    latency_ms: float
    finish_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            # Prompts can be large — exporting them inline blows up the file.
            # Keep length only; full prompts live in the run log.
            "prompt_chars": len(self.prompt),
            "raw_response_preview": self.raw_response[:512],
            "parsed_tool_call": self.parsed_tool_call,
            "observation_preview": (self.observation or "")[:512],
            "latency_ms": round(self.latency_ms, 2),
            "finish_reason": self.finish_reason,
        }


class LLMBackend(Protocol):
    """Tiny protocol the generator relies on. Production backends should
    implement structured tool-calling natively; the stub returns a JSON
    string we parse below."""

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_new_tokens: int = 512,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Stub backend — no external deps, deterministic.
# ---------------------------------------------------------------------------


@dataclass
class StubLLMBackend:
    """Keyword-driven backend used when no real LLM is configured.

    The mapping is intentionally small. Production runs should replace
    this with a real model — but tests, CI, and the example pipeline all
    work without a GPU.
    """

    keyword_to_tool: dict[str, str] = field(
        default_factory=lambda: {
            "moisture": "soil_sensor",
            "soil": "soil_sensor",
            "irrigat": "soil_sensor",
            "weather": "weather_forecast",
            "rain": "weather_forecast",
            "forecast": "weather_forecast",
            "price": "mandi_price",
            "mandi": "mandi_price",
            "market": "mandi_price",
            "pest": "pest_reports",
        }
    )
    default_args: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "soil_sensor": {"lat": 12.97, "lng": 77.59},
            "weather_forecast": {"lat": 12.97, "lng": 77.59, "days": 3},
            "mandi_price": {"commodity": "tomato", "market": "Bengaluru"},
            "pest_reports": {"district": "Mysuru", "days": 7},
        }
    )

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        last_tool = next(
            (m for m in reversed(messages) if m.get("role") == "tool"),
            None,
        )
        n_prior_tool_calls = sum(
            1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
        )

        # If we already saw a tool result, finalise.
        if last_tool is not None and n_prior_tool_calls >= 1:
            return json.dumps(
                {
                    "final": (
                        "Based on the tool results, here is what I found. "
                        f"({last_tool.get('content', '')[:120]})"
                    )
                }
            )

        text = (last_user.get("content") if last_user else "") or ""
        text_lc = text.lower()

        chosen: str | None = None
        if tools:
            registry_tools = {t.get("name") for t in tools if t.get("name")}
            for kw, tool in self.keyword_to_tool.items():
                if kw in text_lc and tool in registry_tools:
                    chosen = tool
                    break
            if chosen is None and registry_tools:
                chosen = sorted(registry_tools)[0]
        else:
            for kw, tool in self.keyword_to_tool.items():
                if kw in text_lc:
                    chosen = tool
                    break

        if chosen is None:
            return json.dumps({"final": "I don't have a tool for that — answering from memory."})

        return json.dumps(
            {
                "tool_call": {
                    "name": chosen,
                    "arguments": dict(self.default_args.get(chosen, {})),
                }
            }
        )


# ---------------------------------------------------------------------------
# Real backends — lazy imports so install stays light.
# ---------------------------------------------------------------------------


@dataclass
class TransformersLLMBackend:
    """transformers.AutoModelForCausalLM driver. Lazy-imports the heavy deps.

    Suitable for single-GPU local runs. For batched H100 generation use
    :class:`VLLMBackend`. The backend just calls ``apply_chat_template`` +
    ``model.generate`` — tool-call structure is parsed downstream by the
    generator's regex.
    """

    model_id: str
    device: str | None = None
    dtype: str = "auto"
    temperature: float = 0.7
    top_p: float = 0.9

    _model: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover — install-time check
            raise ImportError(
                "transformers/torch are not installed. "
                "Install with `pip install training-pipeline[generate]`."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict[str, Any] = {}
        if self.dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, self.dtype)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device:
            self._model = self._model.to(self.device)

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        self._load()
        chat = self._tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(chat, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


@dataclass
class VLLMBackend:
    """vLLM driver for batched generation on multi-GPU boxes.

    The generator's loop is single-trajectory, but vLLM still helps because
    each trajectory step can run with the engine's PagedAttention cache
    primed by the previous step's prompt prefix.
    """

    model_id: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_model_len: int | None = None

    _llm: Any = field(init=False, default=None)
    _sampling: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — install-time check
            raise ImportError(
                "vllm is not installed. Install with `pip install training-pipeline[generate]` "
                "(vLLM is platform-specific; see vLLM docs)."
            ) from exc
        kwargs: dict[str, Any] = {"model": self.model_id}
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)
        self._sampling = SamplingParams(temperature=self.temperature, top_p=self.top_p)
        self._tokenizer = self._llm.get_tokenizer()

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        self._load()
        chat = self._tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling = self._sampling
        if max_new_tokens:
            from vllm import SamplingParams  # type: ignore[import-not-found]

            sampling = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_new_tokens,
            )
        outputs = self._llm.generate([chat], sampling)
        return outputs[0].outputs[0].text


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------


def _find_balanced_object(text: str, key: str) -> dict[str, Any] | None:
    """Find the smallest balanced JSON object in ``text`` that contains ``key``.

    Walks left from the first ``"key"`` occurrence to the nearest ``{``, then
    forward tracking brace depth (string-aware) to the matching ``}``. Avoids
    the false-negatives a flat regex hits when the value itself contains
    ``{`` or ``}``.
    """
    needle = f'"{key}"'
    needle_idx = text.find(needle)
    if needle_idx < 0:
        return None
    open_brace = text.rfind("{", 0, needle_idx)
    if open_brace < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(open_brace, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[open_brace : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                if isinstance(obj, dict) and key in obj:
                    return obj
                return None
    return None


def _parse_response(raw: str) -> dict[str, Any]:
    """Pull a tool_call or final-answer block out of free-form model output.

    Real backends should be configured to emit structured tool calls — but
    we also accept a plain JSON object embedded in prose so the stub
    backend (and noisy production models) work without a function-calling
    wrapper.
    """
    text = raw.strip()
    if not text:
        return {"final": ""}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and ("tool_call" in obj or "final" in obj):
            return obj
    except json.JSONDecodeError:
        pass

    obj = _find_balanced_object(text, "tool_call")
    if obj is not None:
        return obj
    obj = _find_balanced_object(text, "final")
    if obj is not None:
        return obj

    # No structured marker: treat the whole thing as a final answer.
    return {"final": text}


def _registry_to_tools(mock_tools: MockToolRegistry) -> list[dict[str, Any]]:
    """Render the tool registry in OpenAI / chat-template tools format.

    Sufficient for ``apply_chat_template(tools=...)``. Real type maps are
    a TODO once we wire the generator to a function-calling model.
    """
    rendered: list[dict[str, Any]] = []
    for name, spec in mock_tools.registry.tools.items():
        properties: dict[str, dict[str, Any]] = {}
        for prop, ptype in spec.properties.items():
            properties[prop] = {"type": ptype if ptype != "any" else "string"}
        rendered.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(spec.required),
                    },
                },
                # Convenience flat field so backends without nested schema
                # support can still see the name.
                "name": name,
            }
        )
    return rendered


@dataclass
class TrajectoryGenerator:
    """Drive an LLM + mock-tool registry to produce a synthetic trajectory."""

    backend: LLMBackend
    mock_tools: MockToolRegistry
    max_steps: int = 5
    """Mirrors the production budget: ≤4 tool calls plus a final answer."""
    drop_on_invalid_args: bool = True
    """When the model proposes args that fail the registry's schema, drop the
    entire trajectory. Aligns with the mentor's quality > scale stance."""
    system_prompt: str | None = None
    source_label: str = "synthetic"
    seed: int = 0

    def generate(self, seed: Seed | str, *, session_id: str | None = None) -> Trajectory | None:
        query = seed.query if isinstance(seed, Seed) else str(seed)
        sid = session_id or self._session_id(seed)
        domain = seed.domain if isinstance(seed, Seed) else None
        seed_id = seed.seed_id if isinstance(seed, Seed) else None

        events: list[Any] = []
        steps: list[GenerationStep] = []

        ts = datetime.now(tz=timezone.utc).replace(microsecond=0)
        events.append(
            UserEvent(
                event_id="u0",
                session_id=sid,
                timestamp=ts,
                content=query,
                lineage_id=seed_id,
            )
        )
        ts += timedelta(seconds=1)

        chat = self._build_initial_messages(query)
        tools_spec = _registry_to_tools(self.mock_tools)

        finish_reason = "max_steps"
        for step in range(self.max_steps):
            t0 = time.perf_counter()
            raw = self.backend.generate(chat, tools=tools_spec or None)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed = _parse_response(raw)

            if "tool_call" in parsed:
                tc = parsed["tool_call"]
                if not isinstance(tc, dict) or not tc.get("name"):
                    log.debug("dropping trajectory %s: malformed tool_call payload", sid)
                    return None
                args = tc.get("arguments") or {}
                if not isinstance(args, dict):
                    return None
                call = ToolCall(id=f"call_{step}", name=str(tc["name"]), arguments=args)

                arg_issues = validate_tool_call(call, registry=self.mock_tools.registry)
                if any(i.severity == "error" for i in arg_issues) and self.drop_on_invalid_args:
                    log.debug(
                        "dropping trajectory %s: arg-schema issues at step %d", sid, step
                    )
                    return None

                events.append(
                    ToolCallEvent(
                        event_id=f"tc{step}",
                        session_id=sid,
                        timestamp=ts,
                        tool_calls=[call],
                        lineage_id=seed_id,
                    )
                )
                ts += timedelta(seconds=1)

                obs = self.mock_tools.call(call.name, call.arguments, call_index=step)
                events.append(
                    ToolResultEvent(
                        event_id=f"tr{step}",
                        session_id=sid,
                        timestamp=ts,
                        tool_call_id=call.id,
                        name=call.name,
                        content=obs.content,
                        is_error=obs.is_error,
                        lineage_id=seed_id,
                        metadata=(
                            {"failure_mode": obs.failure_mode.value}
                            if obs.failure_mode
                            else {}
                        ),
                    )
                )
                ts += timedelta(seconds=1)

                steps.append(
                    GenerationStep(
                        step=step,
                        prompt=self._stringify(chat),
                        raw_response=raw,
                        parsed_tool_call={"name": call.name, "arguments": call.arguments},
                        observation=obs.content,
                        latency_ms=latency_ms,
                        finish_reason="tool_call",
                    )
                )
                chat.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                        ],
                    }
                )
                chat.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": obs.content,
                    }
                )
                continue

            # Either {"final": "..."} or untagged text — treat as the final.
            content = parsed.get("final") if isinstance(parsed, dict) else None
            content = (content or raw or "").strip()
            events.append(
                AssistantEvent(
                    event_id=f"a{step}",
                    session_id=sid,
                    timestamp=ts,
                    content=content,
                    finish_reason="stop",
                    lineage_id=seed_id,
                )
            )
            steps.append(
                GenerationStep(
                    step=step,
                    prompt=self._stringify(chat),
                    raw_response=raw,
                    parsed_tool_call=None,
                    observation=None,
                    latency_ms=latency_ms,
                    finish_reason="final",
                )
            )
            finish_reason = "final"
            break

        tags = {
            "synthetic": {
                "seed_id": seed_id,
                "n_steps": len(steps),
                "finish_reason": finish_reason,
                "steps": [s.as_dict() for s in steps],
            }
        }
        return Trajectory(
            session_id=sid,
            events=events,
            source=self.source_label,
            domain=domain,
            tags=tags,
            lineage_id=seed_id,
        )

    def generate_many(
        self, seeds: Iterable[Seed | str]
    ) -> Iterable[Trajectory]:
        for s in seeds:
            traj = self.generate(s)
            if traj is not None:
                yield traj

    def generate_to_jsonl(
        self,
        seeds_path: str | Path,
        output_path: str | Path,
    ) -> tuple[int, int]:
        """Stream seeds.jsonl → synthetic_trajectories.jsonl.

        Returns ``(written, dropped)`` so callers can report a yield rate.
        """
        written = 0
        dropped = 0

        def _seeds() -> Iterable[Seed]:
            for rec in iter_records(seeds_path):
                # Tolerate both Seed.as_dict() rows and plain {"query": ...} rows.
                yield Seed(
                    seed_id=rec.get("seed_id") or _stable_id(rec.get("query", "")),
                    query=rec["query"],
                    cluster_id=int(rec.get("cluster_id", 0)),
                    cluster_size=int(rec.get("cluster_size", 1)),
                    original_lineage_ids=list(rec.get("original_lineage_ids") or []),
                    domain=rec.get("domain"),
                    source_session_ids=list(rec.get("source_session_ids") or []),
                )

        def _generated() -> Iterable[Trajectory]:
            nonlocal written, dropped
            for s in _seeds():
                traj = self.generate(s)
                if traj is None:
                    dropped += 1
                    continue
                written += 1
                yield traj

        write_jsonl(output_path, _generated())
        return written, dropped

    def _build_initial_messages(self, query: str) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": query})
        return msgs

    def _session_id(self, seed: Seed | str) -> str:
        if isinstance(seed, Seed):
            return f"syn-{seed.seed_id}-{uuid.uuid4().hex[:6]}"
        return f"syn-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _stringify(messages: Sequence[Mapping[str, Any]]) -> str:
        # Round-tripped only for diagnostic step logs; full chat objects live
        # in run logs.
        return json.dumps(list(messages), default=str)


def _stable_id(query: str) -> str:
    import hashlib

    return f"seed-{hashlib.sha256(query.encode('utf-8')).hexdigest()[:12]}"
