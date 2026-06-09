import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

# =========================
# Load Model
# =========================

MODEL_PATH = "sia_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# =========================
# Input CSV
# =========================

INPUT_FILE = "customer_support_tickets.csv"

df = pd.read_csv(INPUT_FILE)

# =========================
# Create Text
# =========================

df["combined_text"] = (
    "Priority: " + df["Priority_Level"].astype(str)
    + " | Subject: " + df["Ticket_Subject"].astype(str)
    + " | Description: " + df["Ticket_Description"].astype(str)
    + " | Category: " + df["Issue_Category"].astype(str)
    + " | Channel: " + df["Ticket_Channel"].astype(str)
    + " | ResolutionHours: " + df["Resolution_Time_Hours"].astype(str)
    + " | Satisfaction: " + df["Satisfaction_Score"].astype(str)
)

# =========================
# Predict
# =========================

predictions = []

for text in df["combined_text"]:

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    predictions.append(pred)

df["predicted_mismatch"] = predictions

df["prediction_label"] = df[
    "predicted_mismatch"
].map({
    0: "Consistent",
    1: "Mismatch"
})

# =========================
# Save Output
# =========================

df.to_csv(
    "predictions.csv",
    index=False
)

print(
    "Saved: predictions.csv"
)
