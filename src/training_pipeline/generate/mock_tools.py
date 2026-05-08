"""Mock tool registry that powers synthetic trajectory generation.

The mentor's guidance is to *assume* a mock tool environment exists and not
build it as project scope — but the synthetic generator still needs *some*
implementation to talk to. ``MockToolRegistry`` is that implementation: a
small wrapper that pulls schemas from the existing ``ToolRegistry`` and
returns observations for a tool call by one of three mechanisms, in order:

1. A pluggable hook — a Python callable or HTTP endpoint — for projects
   that already have a real mock environment. The protocol is intentionally
   tiny so swapping in an external service is a one-line change.
2. A fixtures directory — JSON files keyed by ``{tool}/{arg_hash}.json``
   give deterministic responses we can commit alongside the dataset.
3. A built-in default that returns a stubbed echo response so a fresh
   install is runnable end-to-end without configuring anything.

Failure injection is a first-class feature, not an afterthought. The
mentor explicitly called out tool failures, partial data, and recovery
trajectories as the high-value edge cases — so the registry can be
configured to raise ``TIMEOUT``, ``INVALID_ARGS``, ``NO_RESULTS``,
``RATE_LIMITED``, or ``PARTIAL_DATA`` for selected tools with a fixed
seed, deterministically, so we can deliberately produce recovery
trajectories without waiting for them to occur naturally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from training_pipeline.validate.consistency import ToolRegistry, ToolSpec

log = logging.getLogger(__name__)


class FailureMode(str, Enum):
    """Deliberate failure modes the registry can inject.

    These cover the four classes of breakage we observe in production logs
    (slow tools, schema drift, empty results, throttling) plus
    ``PARTIAL_DATA`` — the trickiest case for a model to handle, where the
    tool returns a syntactically valid response that's missing a critical
    field.
    """

    TIMEOUT = "TIMEOUT"
    INVALID_ARGS = "INVALID_ARGS"
    NO_RESULTS = "NO_RESULTS"
    RATE_LIMITED = "RATE_LIMITED"
    PARTIAL_DATA = "PARTIAL_DATA"


@dataclass
class ToolResult:
    """One mock tool observation, in a shape we can drop into a Trajectory."""

    tool_name: str
    arguments: dict[str, Any]
    content: str
    """Always a string — that's what ``ToolResultEvent.content`` requires."""
    is_error: bool = False
    failure_mode: FailureMode | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_event_payload(self) -> dict[str, Any]:
        """Compatibility helper: shape the result fits a ``ToolResultEvent``."""
        return {
            "name": self.tool_name,
            "content": self.content,
            "is_error": self.is_error,
        }


# Type aliases for the pluggable hook.
ToolHook = Callable[[str, Mapping[str, Any]], "ToolResult | dict[str, Any] | str"]


def _arg_hash(args: Mapping[str, Any]) -> str:
    """Stable hash of arguments — used to look up fixture files."""
    payload = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _failure_payload(mode: FailureMode, name: str, args: Mapping[str, Any]) -> str:
    """Canonical payload string for each failure mode.

    The strings are intentionally close to what real systems emit so the
    LLM can learn to parse them — not a synthetic ``{error: 'fake'}`` that
    looks nothing like production.
    """
    if mode is FailureMode.TIMEOUT:
        return json.dumps(
            {
                "error": "timeout",
                "message": f"{name} did not return within budget",
            }
        )
    if mode is FailureMode.INVALID_ARGS:
        return json.dumps(
            {
                "error": "invalid_arguments",
                "message": f"argument schema rejected by {name}",
                "received": dict(args),
            }
        )
    if mode is FailureMode.NO_RESULTS:
        return json.dumps({"results": [], "count": 0})
    if mode is FailureMode.RATE_LIMITED:
        return json.dumps(
            {
                "error": "rate_limited",
                "message": "request throttled",
                "retry_after_seconds": 5,
            }
        )
    if mode is FailureMode.PARTIAL_DATA:
        return json.dumps({"partial": True, "fields_present": list(args.keys())})
    raise AssertionError(f"unhandled failure mode: {mode!r}")  # pragma: no cover


def _coerce_result(
    raw: ToolResult | dict[str, Any] | str,
    *,
    tool_name: str,
    args: Mapping[str, Any],
) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    if isinstance(raw, dict):
        content = raw.get("content")
        if not isinstance(content, str):
            content = json.dumps(raw)
        return ToolResult(
            tool_name=tool_name,
            arguments=dict(args),
            content=content,
            is_error=bool(raw.get("is_error", False)),
            metadata={k: v for k, v in raw.items() if k not in {"content", "is_error"}},
        )
    if isinstance(raw, str):
        return ToolResult(
            tool_name=tool_name,
            arguments=dict(args),
            content=raw,
        )
    raise TypeError(f"unsupported tool hook return type: {type(raw).__name__}")


@dataclass
class MockToolRegistry:
    """Mock tool dispatcher with fixtures, a pluggable hook, and failure injection."""

    registry: ToolRegistry = field(default_factory=ToolRegistry)
    fixtures_dir: Path | None = None
    hook: ToolHook | None = None
    """Optional callable invoked as ``hook(tool_name, args)`` for any tool we
    don't have a fixture for. Replaceable with an HTTP-bound shim."""
    failure_config: dict[str, dict[FailureMode, float]] = field(default_factory=dict)
    """Per-tool failure probability map, e.g.
    ``{'mandi_price': {FailureMode.NO_RESULTS: 0.2}}``."""
    seed: int = 0
    default_latency_ms: float = 5.0
    strict_args: bool = True
    """When true, reject calls with arg-schema errors with INVALID_ARGS instead
    of forwarding to the hook."""

    @classmethod
    def from_config(
        cls,
        tool_registry: ToolRegistry | str | Path,
        *,
        fixtures_dir: str | Path | None = None,
        hook: ToolHook | None = None,
        failure_config: Mapping[str, Mapping[str | FailureMode, float]] | None = None,
        seed: int = 0,
    ) -> MockToolRegistry:
        if isinstance(tool_registry, (str, Path)):
            registry = ToolRegistry.from_yaml(tool_registry)
        else:
            registry = tool_registry
        coerced: dict[str, dict[FailureMode, float]] = {}
        if failure_config:
            for name, modes in failure_config.items():
                coerced[name] = {
                    (m if isinstance(m, FailureMode) else FailureMode(m)): float(p)
                    for m, p in modes.items()
                }
        return cls(
            registry=registry,
            fixtures_dir=Path(fixtures_dir) if fixtures_dir else None,
            hook=hook,
            failure_config=coerced,
            seed=seed,
        )

    def has(self, tool_name: str) -> bool:
        return self.registry.has(tool_name)

    def spec(self, tool_name: str) -> ToolSpec | None:
        return self.registry.tools.get(tool_name)

    def call(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        call_index: int = 0,
    ) -> ToolResult:
        """Dispatch one tool call. ``call_index`` is mixed into the failure
        RNG so two identical calls inside the same trajectory don't always
        sample the same failure (useful for testing thrashing recovery)."""
        args_dict = dict(args)
        spec = self.registry.tools.get(tool_name)

        if spec is None and self.registry.tools:
            return self._error(
                FailureMode.INVALID_ARGS,
                tool_name=tool_name,
                args=args_dict,
                detail=f"tool {tool_name!r} not in registry",
            )

        if self.strict_args and spec is not None:
            errs = spec.validate_args(args_dict)
            if errs:
                return self._error(
                    FailureMode.INVALID_ARGS,
                    tool_name=tool_name,
                    args=args_dict,
                    detail="; ".join(errs),
                )

        injected = self._maybe_inject_failure(tool_name, args_dict, call_index=call_index)
        if injected is not None:
            return injected

        if self.fixtures_dir is not None:
            fixture = self._lookup_fixture(tool_name, args_dict)
            if fixture is not None:
                return fixture

        if self.hook is not None:
            try:
                raw = self.hook(tool_name, args_dict)
            except Exception as exc:  # pragma: no cover — hook contract
                log.warning("tool hook for %s raised %s", tool_name, exc)
                return ToolResult(
                    tool_name=tool_name,
                    arguments=args_dict,
                    content=json.dumps({"error": "hook_exception", "message": str(exc)}),
                    is_error=True,
                    latency_ms=self.default_latency_ms,
                )
            res = _coerce_result(raw, tool_name=tool_name, args=args_dict)
            if not res.latency_ms:
                res.latency_ms = self.default_latency_ms
            return res

        # Default: echo the args back as a structured stub. Lets the model and
        # the rest of the pipeline run without any configuration.
        return ToolResult(
            tool_name=tool_name,
            arguments=args_dict,
            content=json.dumps(
                {
                    "tool": tool_name,
                    "args": args_dict,
                    "stub": True,
                }
            ),
            latency_ms=self.default_latency_ms,
        )

    def _lookup_fixture(self, tool_name: str, args: Mapping[str, Any]) -> ToolResult | None:
        if self.fixtures_dir is None:
            return None
        h = _arg_hash(args)
        candidates = [
            self.fixtures_dir / tool_name / f"{h}.json",
            self.fixtures_dir / tool_name / "default.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    log.warning("invalid fixture %s: %s", path, exc)
                    return None
                res = _coerce_result(raw, tool_name=tool_name, args=args)
                if not res.latency_ms:
                    res.latency_ms = self.default_latency_ms
                return res
        return None

    def _maybe_inject_failure(
        self, tool_name: str, args: Mapping[str, Any], *, call_index: int
    ) -> ToolResult | None:
        config = self.failure_config.get(tool_name)
        if not config:
            return None
        rng = random.Random(self._failure_seed(tool_name, args, call_index))
        # Sample modes in declared order so re-arranging the config doesn't
        # silently change the picked mode for the same (seed, args).
        for mode, prob in config.items():
            if prob <= 0:
                continue
            if rng.random() < prob:
                return self._error(mode, tool_name=tool_name, args=args)
        return None

    def _failure_seed(
        self, tool_name: str, args: Mapping[str, Any], call_index: int
    ) -> int:
        h = hashlib.blake2b(
            f"{self.seed}::{tool_name}::{_arg_hash(args)}::{call_index}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(h, "big")

    def _error(
        self,
        mode: FailureMode,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        detail: str | None = None,
    ) -> ToolResult:
        content = _failure_payload(mode, tool_name, args)
        if detail and mode is FailureMode.INVALID_ARGS:
            payload = json.loads(content)
            payload["message"] = detail
            content = json.dumps(payload)
        return ToolResult(
            tool_name=tool_name,
            arguments=dict(args),
            content=content,
            is_error=True,
            failure_mode=mode,
            latency_ms=self.default_latency_ms,
            metadata={"injected": True} if not detail else {"reason": "schema"},
        )
