from dataclasses import dataclass


# Data class representing an AI model configuration
@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
