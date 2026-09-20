import pandas as pd

FILE = "data/processed/xbox_english_sample.csv"
BATCH_SIZE = 100

df = pd.read_csv(FILE)

pd.set_option("display.max_colwidth", 250)

print(f"Total messages: {len(df)}")

while True:
    user_input = input(
        f"\nEnter batch number (1-{(len(df) + BATCH_SIZE - 1) // BATCH_SIZE}) "
        "or 'q' to quit: "
    )

    if user_input.lower() == "q":
        break

    try:
        batch_number = int(user_input)
    except ValueError:
        print("Enter a number or q.")
        continue

    start = (batch_number - 1) * BATCH_SIZE
    end = start + BATCH_SIZE

    if start < 0 or start >= len(df):
        print("Invalid batch number.")
        continue

    batch = df.iloc[start:end]

    print(f"\n{'=' * 80}")
    print(f"BATCH {batch_number}: messages {start + 1}-{min(end, len(df))}")
    print(f"{'=' * 80}")

    for _, row in batch.iterrows():
        print(f"\n[{row['tweet_id']}] {row['text']}")