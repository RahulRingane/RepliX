import pandas as pd

FILE = "data/processed/xbox_english_sample.csv"
BATCH_SIZE = 5

INTENTS = {
    1: "account_access",
    2: "account_security",
    3: "network_connectivity",
    4: "game_issue",
    5: "console_issue",
    6: "controller_accessories",
    7: "audio_video",
    8: "purchase_payment",
    9: "subscription_membership",
    10: "download_update",
    11: "achievements_rewards",
    12: "enforcement",
    13: "order_repair",
    14: "availability_compatibility",
    15: "general_information",
}


def show_intents():
    print("\nIntent options:")
    for number, intent in INTENTS.items():
        print(f"  {number:2}. {intent}")


def save(df):
    df.to_csv(FILE, index=False)


def main():
    df = pd.read_csv(FILE)

    if "intent" not in df.columns:
        df["intent"] = ""

    df["intent"] = df["intent"].astype("string")
    df["intent"] = df["intent"].fillna("")

    total = len(df)

    print(f"\nLoaded {total} messages.")
    print("Enter 5 intent numbers separated by commas.")
    print("Example: 5,4,1,14,3")
    print("Commands: s = skip batch | q = quit")

    while True:

        # Get next 5 unlabeled rows
        batch = []

        for index, row in df.iterrows():
            current = str(row["intent"]).strip()

            if not current or current.lower() == "nan":
                batch.append(index)

            if len(batch) == BATCH_SIZE:
                break

        if not batch:
            print("\nAll messages are labeled.")
            break

        print("\n" + "=" * 100)
        print(
            f"Messages {batch[0] + 1}-{batch[-1] + 1} "
            f"of {total}"
        )
        print("=" * 100)

        for number, index in enumerate(batch, start=1):
            print(f"\n[{number}]")
            print(df.at[index, "text"])

        print("\n" + "=" * 100)
        show_intents()

        while True:
            choice = input(
                f"\nYour choices ({len(batch)} numbers, comma-separated): "
            ).strip().lower()

            if choice == "q":
                save(df)
                print("\nProgress saved. Exiting.")
                return

            if choice == "s":
                print("\nBatch skipped.")
                break

            try:
                choices = [int(x.strip()) for x in choice.split(",")]
            except ValueError:
                print(
                    f"Invalid input. Enter exactly {len(batch)} "
                    "numbers separated by commas."
                )
                continue

            if len(choices) != len(batch):
                print(
                    f"You must enter exactly {len(batch)} numbers."
                )
                continue

            if not all(number in INTENTS for number in choices):
                print("Every number must be between 1 and 15.")
                continue

            # Apply labels
            for index, intent_number in zip(batch, choices):
                df.at[index, "intent"] = INTENTS[intent_number]

            # Save entire batch
            save(df)

            print("\n✓ Batch saved.")

            for position, intent_number in enumerate(choices, start=1):
                print(f"  [{position}] {INTENTS[intent_number]}")

            break


if __name__ == "__main__":
    main()

