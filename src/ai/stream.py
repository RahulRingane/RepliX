from dataclasses import dataclass
from typing import Literal

# Data class representing a stream event type
StreamEventType = Literal[
    "text_delta",
    "tool_call",
    "done",
]


# Data class representing a stream event
@dataclass
class StreamEvent:
    type: StreamEventType
    content: str | None = None
    tool_call_id: str | None = None
