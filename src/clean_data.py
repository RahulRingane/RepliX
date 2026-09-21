import pandas as pd

INPUT_FILE = "data/processed/xbox_english_sample.csv"
OUTPUT_FILE = "data/processed/xbox_english_clean.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only columns needed for intent classification
df = df[["text", "intent"]]

# Remove accidental whitespace
df["text"] = df["text"].str.strip()
df["intent"] = df["intent"].str.strip()

# Remove rows with missing values
df = df.dropna(subset=["text", "intent"])

# Remove empty messages
df = df[df["text"] != ""]

# Remove exact duplicate rows
df = df.drop_duplicates()

df.to_csv(OUTPUT_FILE, index=False)

print(f"Cleaned dataset: {len(df)} rows")
print(f"Saved to: {OUTPUT_FILE}")
print("\nIntent distribution:")
print(df["intent"].value_counts())