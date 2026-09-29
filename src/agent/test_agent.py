import os

from dotenv import load_dotenv

from src.agent.agent import Agent
from src.ai.providers.openai import OpenAIProvider


load_dotenv()


def main():
    provider = OpenAIProvider(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    agent = Agent(
        provider=provider,
        model="gpt-4o-mini",
    )

    response = agent.run(
        "My Xbox is not connecting to the internet."
    )

    print("Response:", response.content)
    print("Messages:", len(agent.state.messages))
    print("Usage:", response.usage)


if __name__ == "__main__":
    main()