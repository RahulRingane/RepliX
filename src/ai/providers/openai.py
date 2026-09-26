from openai import OpenAI
from ..llm_types import LLMRequest, LLMResponse, Usage
from .base import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        response = self.client.responses.create(
            model=request.model,
            input=messages,
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