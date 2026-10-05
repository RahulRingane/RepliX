from dataclasses import dataclass, field

from src.ai.llm_types import Message


# Data class representing the state of an agent, including its message history.
@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
