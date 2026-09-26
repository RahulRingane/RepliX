from abc import ABC, abstractmethod

from ..llm_types import LLMRequest, LLMResponse


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass