"""Evaluate the saved honest single-sequence BERT on the external validation."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = "rubert_salary_local_valid_only"
TARGET = "log_salary_from"

train = pd.read_csv("train.csv")
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
_, valid = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["salary_bin"],
)
valid = valid.drop(columns="salary_bin").reset_index(drop=True)
for column in ["title", "skills", "description", "location", "company"]:
    valid[column] = valid[column].fillna("").astype(str)

texts = (
    "Название: " + valid["title"] + " [SEP] "
    + "Навыки: " + valid["skills"] + " [SEP] "
    + "Описание: " + valid["description"] + " [SEP] "
    + "Регион: " + valid["location"] + " [SEP] "
    + "Компания: " + valid["company"] + " [SEP] "
    + "Опыт от: " + valid["experience_from"].astype(str)
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

predictions = []
with torch.inference_mode():
    for start in range(0, len(texts), 16):
        batch = tokenizer(
            texts.iloc[start : start + 16].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits.reshape(-1)
        predictions.append(logits.float().cpu().numpy())

prediction = np.concatenate(predictions).astype("float32")
y = valid[TARGET].to_numpy(dtype="float32")
print("Second BERT R2:", r2_score(y, prediction))
print(pd.Series(prediction).describe())

output = valid[["title", "location", "company", "experience_from", TARGET]].copy()
output["second_bert_prediction"] = prediction
output.to_csv("local_valid_second_bert_081.csv", index=False, encoding="utf-8-sig")
