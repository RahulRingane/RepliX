from dataclasses import dataclass
from typing import Literal


StreamEventType = Literal[
    "text_delta",
    "tool_call",
    "done",
]


@dataclass
class StreamEvent:
    type: StreamEventType
    content: str | None = None
    tool_call_id: str | None = None