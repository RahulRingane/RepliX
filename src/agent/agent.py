from src.agent.prompts.registry import get_active_prompt
from src.ai.config import LLMConfig
from src.ai.llm_types import LLMRequest, LLMResponse, Message
from src.ai.providers.base import LLMProvider

from .state import AgentState


# Agent class that manages the interaction with the LLM provider,
# maintains the conversation state, and handles prompts.
class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        config: LLMConfig,
        prompt_name: str,
    ):
        self.provider = provider
        self.config = config

        prompt = get_active_prompt(
            prompt_name,
            config.model,
        )
        self.prompt_version = prompt.version
        self.prompt_name = prompt.name

        self.state = AgentState(
            messages=[
                Message(
                    role="system",
                    content=prompt.content,
                )
            ]
        )

    # Run method that takes a user message, appends it to the conversation state,
    # generates a response from the LLM provider, and appends the response to the state.
    def run(self, user_message: str) -> LLMResponse:
        self.state.messages.append(
            Message(
                role="user",
                content=user_message,
            )
        )

        request = LLMRequest(
            model=self.config.model,
            messages=self.state.messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            metadata={
                "agent": self.prompt_name,
                "prompt_name": self.prompt_name,
                "prompt_version": self.prompt_version,
                "provider": self.config.provider,
                "model": self.config.model,
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
