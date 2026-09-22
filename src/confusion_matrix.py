import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

TRAIN_FILE = "data/splits/train.csv"
TEST_FILE = "data/splits/test.csv"

# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["intent"]
y_test = test_df["intent"]

# Train baseline
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

# Display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    xticks_rotation=90,
    values_format="d"
)

plt.tight_layout()
plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved confusion matrix to: models/confusion_matrix.png")