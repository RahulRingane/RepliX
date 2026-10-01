from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.ai.providers.errors import LLMProviderError
from src.ai.providers.openai import OpenAIProvider


# Test successful OpenAI response normalization.
def test_openai_provider_normalizes_response():
    fake_response = SimpleNamespace(
        output_text="Test response",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )

    provider = OpenAIProvider.__new__(OpenAIProvider)

    provider.client = Mock()
    provider.client.responses.create.return_value = fake_response

    request = SimpleNamespace(
        model="gpt-4o-mini",
        messages=[
            SimpleNamespace(
                role="user",
                content="Hello",
            )
        ],
        temperature=0.0,
        max_tokens=100,
    )

    response = provider.generate(request)

    assert response.content == "Test response"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert response.usage.total_tokens == 30
    assert response.metadata["provider"] == "openai"
    assert response.metadata["model"] == "gpt-4o-mini"

    provider.client.responses.create.assert_called_once()


# Test that OpenAI API failures are converted into LLMProviderError.
def test_openai_provider_raises_provider_error():
    provider = OpenAIProvider.__new__(OpenAIProvider)

    provider.client = Mock()
    provider.client.responses.create.side_effect = Exception("API failed")

    request = SimpleNamespace(
        model="gpt-4o-mini",
        messages=[],
        temperature=0.0,
        max_tokens=100,
    )

    with pytest.raises(LLMProviderError, match="OpenAI request failed"):
        provider.generate(request)
