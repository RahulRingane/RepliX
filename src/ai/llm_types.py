from dataclasses import dataclass, field
from typing import Any, Literal
from .cache.types import CacheConfig


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMRequest:
    model: str
    messages: list[Message]

    temperature: float | None = None
    max_tokens: int | None = None

    tools: list[dict[str, Any]] = field(default_factory=list)

    cache: CacheConfig | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    content: str

    tool_calls: list[ToolCall] = field(default_factory=list)

    usage: Usage | None = None

    finish_reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)