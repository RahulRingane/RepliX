import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

FILE = "data/processed/xbox_english_sample.csv"
BATCH_SIZE = 5

INTENTS = [
    "account_access",
    "account_security",
    "network_connectivity",
    "game_issue",
    "console_issue",
    "controller_accessories",
    "audio_video",
    "purchase_payment",
    "subscription_membership",
    "download_update",
    "achievements_rewards",
    "enforcement",
    "order_repair",
    "availability_compatibility",
    "general_information",
]

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def classify_batch(messages):
    formatted = "\n\n".join(
        f"[{i + 1}]\n{message}"
        for i, message in enumerate(messages)
    )

    intent_list = "\n".join(
        f"{i + 1}. {intent}"
        for i, intent in enumerate(INTENTS)
    )

    prompt = f"""
You are an intent classification system for Xbox customer support.

Classify each customer message into EXACTLY ONE intent.

Available intents:

{intent_list}

Rules:
- Return exactly one intent name for each message.
- Preserve the message order.
- Return ONLY the intent names.
- One intent per line.
- Do not add explanations.
- Do not add numbering.

Messages:

{formatted}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a precise customer-support intent classifier.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return [
        line.strip()
        for line in response.choices[0].message.content.strip().splitlines()
        if line.strip()
    ]


def main():
    df = pd.read_csv(FILE)

    if "intent" not in df.columns:
        df["intent"] = ""

    df["intent"] = df["intent"].astype("string").fillna("")

    total = len(df)

    unlabeled = [
        index
        for index, row in df.iterrows()
        if not row["intent"].strip()
    ]

    print(f"Total messages: {total}")
    print(f"Already labeled: {total - len(unlabeled)}")
    print(f"Remaining: {len(unlabeled)}")

    for start in range(0, len(unlabeled), BATCH_SIZE):

        batch_indices = unlabeled[start:start + BATCH_SIZE]

        messages = [
            df.at[index, "text"]
            for index in batch_indices
        ]

        print("\n" + "=" * 80)
        print(
            f"Processing {start + 1}-{start + len(batch_indices)} "
            f"of {len(unlabeled)}"
        )
        print("=" * 80)

        try:
            predictions = classify_batch(messages)

            if len(predictions) != len(batch_indices):
                print("ERROR: Groq returned wrong number of labels.")
                print("Response:", predictions)
                print("Batch skipped.")
                continue

            valid_batch = True

            for prediction in predictions:
                if prediction not in INTENTS:
                    print(f"Invalid intent from Groq: {prediction}")
                    valid_batch = False

            if not valid_batch:
                print("Batch NOT saved.")
                continue

            for index, prediction in zip(batch_indices, predictions):
                df.at[index, "intent"] = prediction
                print(f"[{index + 1}] {prediction}")

            df.to_csv(FILE, index=False)

            print("✓ Batch saved")

            time.sleep(1)

        except Exception as e:
            print(f"ERROR: {e}")
            print("Stopping. Already saved batches are safe.")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
