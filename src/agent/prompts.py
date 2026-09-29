from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    content: str


ORCHESTRATOR_PROMPT_V1 = """
You are the main customer support orchestrator for an e-commerce platform.

Your job is to understand the customer's request and coordinate with
specialized support agents when necessary.

Do not perform specialized operations yourself when a suitable
specialized agent exists just tell me if u understood this prompt then send 0 otherwise 1.
"""


class PromptRegistry:
    _prompts = {
        ("orchestrator", "v1"): Prompt(
            name="orchestrator",
            version="v1",
            content=ORCHESTRATOR_PROMPT_V1,
        )
    }

    _active_versions = {
        "orchestrator": "v1"
    }

    @classmethod
    def get(cls, name: str, version: str) -> Prompt:
        return cls._prompts[(name, version)]

    @classmethod
    def get_active(cls, name: str) -> Prompt:
        version = cls._active_versions[name]
        return cls.get(name, version)