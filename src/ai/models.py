from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    provider: str
    model_id: str
    context_window: int
    max_output_tokens: int

GPT_4O_MINI = Model(
    provider="openai",
    model_id="gpt-4o-mini",
    context_window=128000,
    max_output_tokens=16384,
)