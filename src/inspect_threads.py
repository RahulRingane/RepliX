import pandas as pd

DATA_PATH = "data/raw/twcs/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "Delta",
    "XboxSupport",
]

df = pd.read_csv(DATA_PATH)

df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"], errors="coerce"
)

# Only tweets belonging to the selected brands
results = []

for brand in BRANDS:

    brand_df = df[df["author_id"] == brand].copy()

    # Brand replies that have a known parent tweet
    replies = brand_df[
        brand_df["in_response_to_tweet_id"].notna()
    ].copy()

    # Group each conversation by the root/starting tweet.
    # Walk backwards through parent IDs.
    tweet_parent = dict(
        zip(df["tweet_id"], df["in_response_to_tweet_id"])
    )

    def get_root(tweet_id):
        current = tweet_id
        visited = set()

        while (
            current in tweet_parent
            and pd.notna(tweet_parent[current])
            and current not in visited
        ):
            visited.add(current)
            current = tweet_parent[current]

        return current

    replies["root_id"] = replies["tweet_id"].apply(get_root)

    # Number of unique conversation threads
    unique_threads = replies["root_id"].nunique()

    # Conversation depth
    thread_sizes = (
        df[df["tweet_id"].isin(
            replies["tweet_id"].tolist()
        )]
        .groupby(
            replies.set_index("tweet_id")
            .loc[
                df[df["tweet_id"].isin(replies["tweet_id"])]["tweet_id"],
                "root_id"
            ]
        )
        .size()
    )

    avg_depth = thread_sizes.mean() if len(thread_sizes) else 0

    # Threads where the brand responded more than once
    brand_replies_per_thread = replies.groupby("root_id").size()

    multi_turn_threads = (
        (brand_replies_per_thread >= 2).sum()
    )

    multi_turn_rate = (
        multi_turn_threads / unique_threads * 100
        if unique_threads
        else 0
    )

    # Actionable language heuristic
    actionable_pattern = (
        r"\bplease\b|"
        r"\btry\b|"
        r"\bcheck\b|"
        r"\bcontact\b|"
        r"\bvisit\b|"
        r"\bcall\b|"
        r"\bdm\b|"
        r"\blink\b|"
        r"\bsteps\b|"
        r"\bclick\b|"
        r"\bupdate\b|"
        r"\bchange\b|"
        r"\breset\b"
    )

    actionable = replies["text"].str.contains(
        actionable_pattern,
        case=False,
        regex=True,
        na=False
    )

    actionable_rate = actionable.mean() * 100

    results.append({
        "brand": brand,
        "unique_threads": unique_threads,
        "avg_brand_replies_per_thread": round(
            brand_replies_per_thread.mean(), 2
        ),
        "multi_turn_thread_rate": round(
            multi_turn_rate, 2
        ),
        "actionable_reply_rate": round(
            actionable_rate, 2
        ),
    })


results_df = pd.DataFrame(results)

print("\n========== FINAL BRAND COMPARISON ==========\n")
print(results_df.to_string(index=False))