import pytest

from src.ai.config import LLMConfig


# Valid configuration
def test_llm_config_valid():
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
    )

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.max_tokens == 256

    # Provider validation


def test_llm_config_rejects_empty_provider():
    with pytest.raises(ValueError, match="Provider cannot be empty"):
        LLMConfig(
            provider="",
            model="gpt-4o-mini",
        )

        # Model validation


def test_llm_config_rejects_empty_model():
    with pytest.raises(ValueError, match="Model cannot be empty"):
        LLMConfig(
            provider="openai",
            model="",
        )

        # Temperature validation


def test_llm_config_rejects_negative_temperature():
    with pytest.raises(ValueError, match="Temperature cannot be negative"):
        LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=-1,
        )

        # Token limit validation


def test_llm_config_rejects_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            max_tokens=0,
        )
