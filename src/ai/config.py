from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
