from openai import OpenAI

from ..llm_types import LLMRequest, LLMResponse, Usage
from .base import LLMProvider
from .errors import LLMProviderError


# OpenAI implementation of the generic LLM provider interface.
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    # Sends the request to OpenAI and converts the response into LLMResponse.
    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        try:
            # Call the OpenAI Responses API.
            response = self.client.responses.create(
                model=request.model,
                input=messages,
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )

            usage = None

            if response.usage:
                usage = Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                )

                return LLMResponse(
                    content=response.output_text,
                    usage=usage,
                    metadata={
                        "provider": "openai",
                        "model": request.model,
                    },
                )

        except Exception as exc:
            # Convert provider-specific errors into a common application error.
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc
