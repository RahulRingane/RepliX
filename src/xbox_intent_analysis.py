import pandas as pd
from collections import Counter
import re

DATA_PATH = "data/raw/twcs/twcs.csv"

df = pd.read_csv(DATA_PATH)

# Get XboxSupport replies
replies = df[
    (df["author_id"] == "XboxSupport") &
    (df["in_response_to_tweet_id"].notna())
].copy()

customer_ids = pd.to_numeric(
    replies["in_response_to_tweet_id"],
    errors="coerce"
).dropna().astype("int64")

customers = df[
    df["tweet_id"].isin(customer_ids)
].copy()

customers = customers[
    customers["author_id"] != "XboxSupport"
]

# Simple English filter
from langdetect import detect, LangDetectException

def is_english(text):
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False

customers["is_english"] = customers["text"].apply(is_english)

customers = customers[customers["is_english"]].copy()

print("\n========== XBOXSUPPORT ENGLISH CUSTOMER MESSAGES ==========\n")
print("Total:", len(customers))

# Look for common support-related keywords
keywords = {
    "account": r"\b(account|login|sign.?in|password|email|gamertag)\b",
    "purchase/payment": r"\b(buy|purchase|payment|pay|charged|refund|money|billing)\b",
    "game": r"\b(game|games|gameplay|fifa|minecraft|halo|fortnite)\b",
    "console": r"\b(xbox|console|xbox one|series x|series s)\b",
    "controller": r"\b(controller|joystick|gamepad)\b",
    "network": r"\b(wifi|wi-fi|internet|network|connection|connect)\b",
    "download/install": r"\b(download|install|installation|update)\b",
    "subscription": r"\b(game pass|gold|subscription|membership)\b",
    "order/shipping": r"\b(order|pre.?order|shipping|delivery|deliver)\b",
    "error/problem": r"\b(error|issue|problem|broken|not working|can't|cannot)\b",
}

print("\n========== KEYWORD COVERAGE ==========\n")

for category, pattern in keywords.items():
    count = customers["text"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).sum()

    percentage = count / len(customers) * 100

    print(f"{category:20} {count:6} ({percentage:5.2f}%)")


print("\n========== RANDOM CUSTOMER EXAMPLES ==========\n")

sample = customers.sample(
    n=min(50, len(customers)),
    random_state=42
)

for i, text in enumerate(sample["text"], 1):
    print(f"{i}. {text}")