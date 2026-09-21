import pandas as pd

files = {
    "Train": "data/splits/train.csv",
    "Validation": "data/splits/validation.csv",
    "Test": "data/splits/test.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)

    print(f"\n{name}: {len(df)} rows")
    print(df["intent"].value_counts().sort_index())