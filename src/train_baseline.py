import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

TRAIN_FILE = "data/splits/train.csv"
VAL_FILE = "data/splits/validation.csv"
TEST_FILE = "data/splits/test.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000
)

X_train = vectorizer.fit_transform(train_df["text"])
X_val = vectorizer.transform(val_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["intent"]
y_val = val_df["intent"]
y_test = test_df["intent"]

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Validation
val_predictions = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("\nValidation Report:")
print(classification_report(y_val, val_predictions))

# Test
test_predictions = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, test_predictions))
print("\nTest Report:")
print(classification_report(y_test, test_predictions))