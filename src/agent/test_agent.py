import os

from dotenv import load_dotenv

from src.agent.agent import Agent
from src.ai.config import LLMConfig
from src.ai.providers.openai import OpenAIProvider

load_dotenv()


def main():
    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])

    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100,
    )

    agent = Agent(
        provider=provider,
        config=config,
        prompt_name="orchestrator",
    )

    response = agent.run("My Xbox is not connecting to the internet.")

    print("Response:", response.content)
    print("Messages:", len(agent.state.messages))
    print("Usage:", response.usage)


if __name__ == "__main__":
    main()
