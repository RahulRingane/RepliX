from src.ai.llm_types import LLMRequest, LLMResponse, Message
from src.ai.providers.base import LLMProvider

from .state import AgentState


class Agent:

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        system_prompt: str,
        prompt_version: str,
    ):
        self.provider = provider
        self.model = model
        self.prompt_version = prompt_version

        self.state = AgentState(
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                )
            ]
        )

    def run(self, user_message: str) -> LLMResponse:
        self.state.messages.append(
            Message(
                role="user",
                content=user_message,
            )
        )

        request = LLMRequest(
            model=self.model,
            messages=self.state.messages,
            metadata={
                "agent": "orchestrator",
                "prompt_version": self.prompt_version,
            },
        )

        response = self.provider.generate(request)

        self.state.messages.append(
            Message(
                role="assistant",
                content=response.content,
            )
        )

        return response