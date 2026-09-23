import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


TRAIN_FILE = "data/splits/train.csv"
VAL_FILE = "data/splits/validation.csv"
TEST_FILE = "data/splits/test.csv"

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "models/bert_intent_classifier"


# -----------------------------
# 1. Load datasets
# -----------------------------

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)


# -----------------------------
# 2. Create label mappings
# -----------------------------

labels = sorted(train_df["intent"].unique())

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

train_df["label"] = train_df["intent"].map(label2id)
val_df["label"] = val_df["intent"].map(label2id)
test_df["label"] = test_df["intent"].map(label2id)


# -----------------------------
# 3. Convert to Hugging Face Dataset
# -----------------------------

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)


# -----------------------------
# 4. Load tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


# -----------------------------
# 5. Load pre-trained BERT
# -----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


# -----------------------------
# 6. Evaluation metrics
# -----------------------------

def compute_metrics(eval_pred):
    predictions, labels_true = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels_true, predictions)

    macro_f1 = f1_score(
        labels_true,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        labels_true,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# -----------------------------
# 7. Training configuration
# -----------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",

    logging_steps=20,

    report_to="none",
)


# -----------------------------
# 8. Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,
)


# -----------------------------
# 9. Train
# -----------------------------

print("Starting BERT training...")

trainer.train()


# -----------------------------
# 10. Evaluate on test set
# -----------------------------

print("\nEvaluating on test set...")

test_results = trainer.evaluate(test_dataset)

print("\nTest Results:")

for key, value in test_results.items():
    print(f"{key}: {value}")


# -----------------------------
# 11. Save model
# -----------------------------

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")