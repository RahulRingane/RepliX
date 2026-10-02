import os

from dotenv import load_dotenv

from src.ai.llm_types import LLMRequest, Message
from src.ai.providers.openai import OpenAIProvider

load_dotenv()


# Example usage of OpenAIProvider to generate a response from the GPT-4o-mini model.
def main():
    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])

    request = LLMRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="Say hello in one sentence.")],
    )

    response = provider.generate(request)

    print("Response:", response.content)
    print("Usage:", response.usage)


if __name__ == "__main__":
    main()
