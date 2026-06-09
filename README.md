# Support Integrity Auditor (SIA)

## Project Overview

Support Integrity Auditor (SIA) is an AI-powered system designed to detect priority mismatches in customer support tickets. The system identifies situations where the assigned ticket priority may not accurately reflect the actual severity of the issue.

The model classifies tickets into:

- Consistent
- Mismatch

Mismatch cases are further categorized as:

- Hidden Crisis (severity underestimated)
- False Alarm (severity overestimated)

---

## Problem Statement

Customer support teams often assign incorrect priorities to incoming tickets. This can lead to:

- Critical issues being ignored or delayed
- Low-severity issues consuming excessive resources
- Reduced customer satisfaction
- Inefficient support operations

The objective of this project is to automatically detect such inconsistencies using machine learning and natural language processing.

---

## Methodology

### Stage 1: Pseudo-Label Generation

A rule-based scoring framework was designed to generate pseudo-labels.

Signals used:

- Priority Level
- Issue Category
- Ticket Subject
- Ticket Description
- Resolution Time
- Customer Satisfaction Score

A final severity score is computed and compared against the assigned priority score.

The system generates:

- `mismatch = 1` → Hidden Crisis / False Alarm
- `mismatch = 0` → Consistent

---

### Stage 2: Text Representation

A rich textual representation is created for each ticket:

- Priority
- Subject
- Description
- Category
- Channel
- Resolution Time
- Satisfaction Score

These fields are combined into a single text sequence for transformer training.

---

### Stage 3: Model Training

Model:

- DistilBERT Base Uncased

Task:

- Binary Classification
- Labels:
  - 0 = Consistent
  - 1 = Mismatch

Training Configuration:

- Epochs: 3
- Learning Rate: 2e-5
- Batch Size: 8
- Weight Decay: 0.01

---

## Evaluation Metrics

The following metrics are used:

- Accuracy
- Macro F1 Score
- Recall

Project verification thresholds:

- Accuracy ≥ 83%
- Macro F1 ≥ 0.82
- Recall ≥ 0.78

---

## Results

Final model performance:

| Metric | Score |
|----------|----------|
| Accuracy | 1.00 |
| Macro F1 | 1.00 |
| Recall | 1.00 |

---

## Repository Structure

```text
.
├── Notebook.ipynb
├── train_pipeline.py
├── predict.py
├── requirements.txt
├── customer_support_tickets.csv
├── pseudo_labeled_dataset.csv
└── README.md
```

---

## How to Train

```bash
python train_pipeline.py
```

The trained model will be saved in:

```text
sia_model/
```
## Trained Model

The trained DistilBERT model is available at:

https://drive.google.com/drive/folders/1_epY4js5yXygUMxrcoX5tuPZIvVje0St?usp=sharing

---

## How to Run Inference

```bash
python predict.py
```

Output:

```text
predictions.csv
```

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Author

Arun Meena  
IIT Roorkee
