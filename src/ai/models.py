from dataclasses import dataclass


# Data class representing an AI model
@dataclass(frozen=True)
class Model:
    provider: str
    model_id: str
    context_window: int
    max_output_tokens: int


# Define GPT_4O_MINI AI model with their properties
GPT_4O_MINI = Model(
    provider="openai",
    model_id="gpt-4o-mini",
    context_window=128000,
    max_output_tokens=16384,
)
