class Agent:
    def __init__(
        self,
        provider,
        model,
        prompt_name,
    ):
        self.provider = provider
        self.model = model

        prompt = PromptRegistry.get_active(prompt_name)

        self.system_prompt = prompt.content
        self.prompt_version = prompt.version
