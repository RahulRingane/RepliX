from dataclasses import dataclass, field

from src.ai.llm_types import Message


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)