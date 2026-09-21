import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed/xbox_english_clean.csv"

TRAIN_FILE = "data/splits/train.csv"
VAL_FILE = "data/splits/validation.csv"
TEST_FILE = "data/splits/test.csv"

df = pd.read_csv(INPUT_FILE)

# 80% train, 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["intent"],
    random_state=42,
)

# Split remaining 20% equally -> 10% validation, 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["intent"],
    random_state=42,
)

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")