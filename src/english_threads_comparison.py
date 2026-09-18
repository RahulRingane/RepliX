import pandas as pd
from langdetect import detect, LangDetectException

DATA_PATH = "data/raw/twcs/twcs.csv"

df = pd.read_csv(DATA_PATH)

def is_english(text):
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


def analyze_brand(brand):
    # Brand replies
    brand_replies = df[
        (df["author_id"] == brand) &
        (df["in_response_to_tweet_id"].notna())
    ].copy()

    # Customer tweets that the brand replied to
    customer_ids = pd.to_numeric(
        brand_replies["in_response_to_tweet_id"],
        errors="coerce"
    ).dropna().astype("int64")

    customers = df[
        df["tweet_id"].isin(customer_ids)
    ].copy()

    customers = customers[
        customers["author_id"] != brand
    ]

    # English customer messages
    customers["english"] = customers["text"].apply(is_english)
    english = customers[customers["english"]].copy()

    print(f"\n========== {brand} ==========")
    print(f"Customer messages: {len(customers)}")
    print(f"English messages:  {len(english)}")
    print(
        f"English %:        "
        f"{len(english) / len(customers) * 100:.2f}%"
    )

    return english


amazon = analyze_brand("AmazonHelp")
xbox = analyze_brand("XboxSupport")

print("\n========== COMPARISON ==========\n")

print(f"AmazonHelp English:  {len(amazon)}")
print(f"XboxSupport English: {len(xbox)}")