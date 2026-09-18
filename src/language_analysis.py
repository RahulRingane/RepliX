import pandas as pd
from langdetect import detect, LangDetectException

DATA_PATH = "data/raw/twcs/twcs.csv"

df = pd.read_csv(DATA_PATH)

# Get tweets that XboxSupport replied to
xbox_replies = df[
    (df["author_id"] == "XboxSupport") &
    (df["in_response_to_tweet_id"].notna())
].copy()

# Get the original customer tweets
customer_ids = pd.to_numeric(
    xbox_replies["in_response_to_tweet_id"],
    errors="coerce"
).dropna().astype("int64")

customers = df[
    df["tweet_id"].isin(customer_ids)
].copy()

# Keep non-XboxSupport tweets only
customers = customers[
    customers["author_id"] != "XboxSupport"
]

sample = customers.sample(
    n=min(10000, len(customers)),
    random_state=42
)

def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"

sample["language"] = sample["text"].apply(detect_language)

print("\n========== XBOXSUPPORT CUSTOMER LANGUAGE DISTRIBUTION ==========\n")

counts = sample["language"].value_counts()
percentages = sample["language"].value_counts(normalize=True) * 100

result = pd.DataFrame({
    "tweets": counts,
    "percentage": percentages.round(2)
})

print(result)