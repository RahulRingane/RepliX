import pandas as pd
from langdetect import detect, LangDetectException
from collections import Counter
import re

DATA_PATH = "data/raw/twcs/twcs.csv"

df = pd.read_csv(DATA_PATH)

# XboxSupport replies
replies = df[
    (df["author_id"] == "XboxSupport") &
    (df["in_response_to_tweet_id"].notna())
]

# Customer tweets that XboxSupport replied to
customer_ids = pd.to_numeric(
    replies["in_response_to_tweet_id"],
    errors="coerce"
).dropna().astype("int64")

customers = df[df["tweet_id"].isin(customer_ids)].copy()
customers = customers[customers["author_id"] != "XboxSupport"]


def is_english(text):
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False


customers["is_english"] = customers["text"].apply(is_english)
customers = customers[customers["is_english"]].copy()

print(f"\nEnglish customer messages: {len(customers)}")


# Common support terms
patterns = {
    "account": r"\b(account|login|sign.?in|password|email|gamertag)\b",
    "network": r"\b(network|internet|wifi|wi-fi|connection|connect|disconnect|online)\b",
    "game": r"\b(game|games|gameplay|multiplayer)\b",
    "console": r"\b(xbox|console|series x|series s|xbox one)\b",
    "controller": r"\b(controller|gamepad|joystick)\b",
    "purchase": r"\b(buy|bought|purchase|payment|pay|charged|refund|billing)\b",
    "subscription": r"\b(game pass|xbox live|gold|subscription|membership|trial)\b",
    "download": r"\b(download|install|installation|update|updat(e|ing))\b",
    "order": r"\b(order|pre.?order|shipping|delivery|deliver)\b",
    "ban": r"\b(ban|banned|suspend|suspended|enforcement)\b",
    "error": r"\b(error|issue|problem|broken|not working|can't|cannot)\b",
    "repair": r"\b(repair|fix|warranty|replacement)\b",
    "compatibility": r"\b(backwards compatible|backward compatible|compatibility)\b",
}

print("\n========== ISSUE COVERAGE ==========\n")

for category, pattern in patterns.items():
    count = customers["text"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).sum()

    percentage = count / len(customers) * 100

    print(f"{category:18} {count:6} ({percentage:5.2f}%)")


# Most common individual words
print("\n========== COMMON WORDS ==========\n")

stopwords = {
    "the", "and", "to", "a", "i", "is", "it", "of", "for",
    "on", "in", "my", "this", "that", "with", "you", "xbox",
    "me", "can", "have", "please", "be", "but", "not", "do",
    "what", "why", "how", "we", "was", "are", "or", "so"
}

counter = Counter()

for text in customers["text"]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())

    for word in words:
        if word not in stopwords:
            counter[word] += 1

for word, count in counter.most_common(50):
    print(f"{word:20} {count}")