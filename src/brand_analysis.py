import pandas as pd

FILE = "data/raw/twcs/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "XboxSupport",
    "airtel_care",
]

print("Loading dataset...")

df = pd.read_csv(FILE)

print(f"Total tweets: {len(df):,}")

# Make IDs numeric
df["tweet_id"] = pd.to_numeric(
    df["tweet_id"],
    errors="coerce"
)

df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

# --------------------------------------------------
# Parent tweet lookup
# --------------------------------------------------

tweet_authors = df[
    ["tweet_id", "author_id"]
].rename(
    columns={
        "author_id": "parent_author_id"
    }
)

results = []

for brand in BRANDS:

    print(f"Analyzing {brand}...")

    # -----------------------------------------------
    # All tweets from this brand
    # -----------------------------------------------

    brand_df = df[
        df["author_id"] == brand
    ]

    # -----------------------------------------------
    # Brand tweets that are replies
    # -----------------------------------------------

    brand_replies = brand_df[
        brand_df["in_response_to_tweet_id"].notna()
    ]

    # -----------------------------------------------
    # Find who the brand replied to
    # -----------------------------------------------

    brand_replies_with_parent = brand_replies.merge(
        tweet_authors,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        how="left",
        suffixes=("", "_parent")
    )

    # Parent was a customer
    customer_replies = brand_replies_with_parent[
        brand_replies_with_parent["parent_author_id"].notna()
        & (
            brand_replies_with_parent["parent_author_id"]
            != brand
        )
    ]

    # -----------------------------------------------
    # DM / redirect detection
    # -----------------------------------------------

    redirect_pattern = (
        r"\bdm\b|"
        r"direct message|"
        r"private message|"
        r"send us a message|"
        r"message us|"
        r"contact us"
    )

    redirect_mask = customer_replies["text"].str.contains(
        redirect_pattern,
        case=False,
        regex=True,
        na=False
    )

    redirect_replies = customer_replies[
        redirect_mask
    ]

    substantive_replies = customer_replies[
        ~redirect_mask
    ]

    # -----------------------------------------------
    # Customer replies after a brand reply
    # -----------------------------------------------

    brand_reply_ids = brand_replies[
        "tweet_id"
    ].dropna()

    followup_customer = df[
        df["in_response_to_tweet_id"].isin(
            brand_reply_ids
        )
        & (df["author_id"] != brand)
    ]

    # -----------------------------------------------
    # Store results
    # -----------------------------------------------

    results.append({
        "brand": brand,

        "brand_tweets": len(
            brand_df
        ),

        "replies_to_customers": len(
            customer_replies
        ),

        "redirect_replies": len(
            redirect_replies
        ),

        "substantive_replies": len(
            substantive_replies
        ),

        "customer_followups": len(
            followup_customer
        ),

        "redirect_rate": (
            len(redirect_replies)
            / len(customer_replies)
            if len(customer_replies) > 0
            else 0
        ),

        "substantive_rate": (
            len(substantive_replies)
            / len(customer_replies)
            if len(customer_replies) > 0
            else 0
        ),

        "followup_rate": (
            len(followup_customer)
            / len(customer_replies)
            if len(customer_replies) > 0
            else 0
        ),
    })


# --------------------------------------------------
# Final results
# --------------------------------------------------

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "substantive_replies",
    ascending=False
)

print("\n========== SUPPORT ANALYSIS ==========\n")

print(
    result_df.to_string(
        index=False,
        formatters={
            "redirect_rate": "{:.2%}".format,
            "substantive_rate": "{:.2%}".format,
            "followup_rate": "{:.2%}".format,
        }
    )
)

result_df.to_csv(
    "data/brand_analysis.csv",
    index=False
)

print(
    "\nSaved to data/brand_analysis.csv"
)