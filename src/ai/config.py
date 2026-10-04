from dataclasses import dataclass


# Data class representing an AI model configuration.
@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self):
        if not self.provider:
            raise ValueError("Provider cannot be empty.")

        if not self.model:
            raise ValueError("Model cannot be empty.")

        if self.temperature < 0:
            raise ValueError("Temperature cannot be negative.")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
