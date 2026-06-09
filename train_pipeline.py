import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("pseudo_labeled_dataset.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["mismatch"]
)

train_df["text"] = train_df["combined_text"]
test_df["text"] = test_df["combined_text"]

# =========================
# HuggingFace Dataset
# =========================

train_dataset = Dataset.from_pandas(
    train_df[["text", "mismatch"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "mismatch"]]
)

# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset = train_dataset.rename_column(
    "mismatch",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "mismatch",
    "labels"
)

train_dataset = train_dataset.remove_columns(
    ["text", "__index_level_0__"]
)

test_dataset = test_dataset.remove_columns(
    ["text", "__index_level_0__"]
)

# =========================
# Model
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# =========================
# Metrics
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro")
    }

# =========================
# Training Arguments
# =========================

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

# =========================
# Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# =========================
# Train
# =========================

trainer.train()

# =========================
# Evaluate
# =========================

metrics = trainer.evaluate()

print(metrics)

# =========================
# Save Model
# =========================

trainer.save_model("sia_model")
tokenizer.save_pretrained("sia_model")
