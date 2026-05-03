"""SFT and DPO export record schemas.

These mirror what the major trainers (TRL ``SFTTrainer`` / ``DPOTrainer``,
HF ``datasets``) accept, with extra ``metadata`` carried alongside for
stratified sampling at training time.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SFTToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SFTMessage(BaseModel):
    """A single message in an SFT chat template.

    Mirrors the OpenAI / Anthropic chat shape so the same JSONL works with
    most trainers and tokenisers.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[SFTToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _validate_role_specific(self) -> SFTMessage:
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
        if self.role == "assistant":
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant messages need content or tool_calls")
        if self.role in ("system", "user") and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        return self


class SFTRecord(BaseModel):
    """One row of an SFT JSONL export."""

    model_config = ConfigDict(extra="forbid")

    messages: list[SFTMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_empty(self) -> SFTRecord:
        if len(self.messages) < 2:
            raise ValueError("SFT record needs at least 2 messages (e.g. user + assistant)")
        return self


class DPORecord(BaseModel):
    """One row of a DPO JSONL export.

    The ``prompt`` is a list of messages so multi-turn preference data fits;
    ``chosen`` and ``rejected`` are completion suffixes (one or more assistant
    turns each).
    """

    model_config = ConfigDict(extra="forbid")

    prompt: list[SFTMessage]
    chosen: list[SFTMessage]
    rejected: list[SFTMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_shape(self) -> DPORecord:
        if not self.prompt:
            raise ValueError("DPO record requires non-empty prompt")
        if not self.chosen or not self.rejected:
            raise ValueError("DPO record requires non-empty chosen and rejected")
        if all(m.role != "assistant" for m in self.chosen):
            raise ValueError("DPO chosen must contain at least one assistant message")
        if all(m.role != "assistant" for m in self.rejected):
            raise ValueError("DPO rejected must contain at least one assistant message")
        return self
