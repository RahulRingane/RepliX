import os
import pandas as pd
from langdetect import detect, LangDetectException

DATA_PATH = "data/raw/twcs/twcs.csv"
OUTPUT_PATH = "data/processed/xbox_english_sample.csv"

SAMPLE_SIZE = 1500


def is_english(text):
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert IDs to nullable integers so both columns have the same format
df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce").astype("Int64")
df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
).astype("Int64")

# 1. Get XboxSupport replies
xbox_replies = df[
    (df["author_id"] == "XboxSupport") &
    (df["in_response_to_tweet_id"].notna())
].copy()

print(f"XboxSupport replies: {len(xbox_replies)}")

# 2. Get IDs of customer tweets XboxSupport replied to
customer_ids = set(
    xbox_replies["in_response_to_tweet_id"].dropna()
)

# 3. Find those customer tweets
customers = df[
    df["tweet_id"].isin(customer_ids) &
    (df["author_id"] != "XboxSupport")
].copy()

print(f"Customer messages: {len(customers)}")

# 4. Keep English messages
print("Detecting language...")

customers["is_english"] = customers["text"].apply(is_english)

english = customers[
    customers["is_english"]
].copy()

english.drop(columns=["is_english"], inplace=True)

print(f"English customer messages: {len(english)}")

# 5. Remove duplicate messages
english = english.drop_duplicates(subset=["text"])

print(f"Unique English messages: {len(english)}")

# 6. Random representative sample
sample_size = min(SAMPLE_SIZE, len(english))

sample = english.sample(
    n=sample_size,
    random_state=42
).copy()

# 7. Keep useful columns
sample = sample[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ]
]

# 8. Empty column for manual intent labeling
sample["intent"] = ""

# 9. Create output directory
os.makedirs("data/processed", exist_ok=True)

# 10. Save
sample.to_csv(OUTPUT_PATH, index=False)

print()
print(f"Saved {len(sample)} messages to:")
print(OUTPUT_PATH)