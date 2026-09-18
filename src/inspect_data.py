import pandas as pd
from pathlib import Path

data_dir = Path("data/raw")

csv_files = list(data_dir.rglob("*.csv"))

print("CSV files found:")
for file in csv_files:
    print(" -", file)

file = data_dir / "twcs" / "twcs.csv"

print(f"\nLoading: {file}")

df = pd.read_csv(file)

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())