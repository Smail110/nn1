"""Train full-data raw-target ruBERT matching the honest validation recipe."""

from __future__ import annotations

import gc
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET = "log_salary_from"
RANDOM_STATE = 42
BASE_MODEL = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--DeepPavlov--rubert-base-cased"
    / "snapshots"
    / "4036cab694767a299f2b9e6492909664d9414229"
)
OUTPUT_DIR = Path("rubert_salary_full_raw_081")
TEST_PREDICTIONS = Path("test_full_raw_bert_081.csv")

MAX_LENGTH = 512
PREFIX_MAX_TOKENS = 160
DESC_CHUNK_TOKENS = 320
DESC_CHUNK_OVERLAP = 80
MAX_CHUNKS_PER_ROW = 6
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
EPOCHS = 4
LEARNING_RATE = 1e-5


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["title", "location", "company", "skills", "description"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["experience_from"] = pd.to_numeric(out["experience_from"], errors="coerce").fillna(0)
    return out


def make_prefix(row: pd.Series) -> str:
    return (
        "Название: " + row["title"] + " [SEP] "
        "Навыки: " + row["skills"] + " [SEP] "
        "Регион: " + row["location"] + " [SEP] "
        "Компания: " + row["company"] + " [SEP] "
        "Опыт от: " + str(row["experience_from"]) + " [SEP] "
        "Описание: "
    )


def description_chunks(tokenizer, description: str):
    ids = tokenizer(str(description), add_special_tokens=False)["input_ids"]
    if not ids:
        return [[]]
    result = []
    start = 0
    while start < len(ids) and len(result) < MAX_CHUNKS_PER_ROW:
        result.append(ids[start : start + DESC_CHUNK_TOKENS])
        if start + DESC_CHUNK_TOKENS >= len(ids):
            break
        start += DESC_CHUNK_TOKENS - DESC_CHUNK_OVERLAP
    return result


def build_chunks(df: pd.DataFrame, tokenizer, include_labels: bool) -> pd.DataFrame:
    result = []
    for row_id, row in df.reset_index(drop=True).iterrows():
        prefix_ids = tokenizer(
            make_prefix(row),
            add_special_tokens=False,
            truncation=True,
            max_length=PREFIX_MAX_TOKENS,
        )["input_ids"]
        for chunk_id, desc_ids in enumerate(description_chunks(tokenizer, row["description"])):
            ids = (prefix_ids + desc_ids)[: MAX_LENGTH - 2]
            item = {
                "row_id": row_id,
                "chunk_id": chunk_id,
                "text": tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
            }
            if include_labels:
                item["label"] = float(row[TARGET])
            result.append(item)
    return pd.DataFrame(result)


class ChunkDataset(Dataset):
    def __init__(self, frame, tokenizer, labels: bool):
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = frame["label"].to_numpy(dtype="float32") if labels else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def aggregate(frame: pd.DataFrame, predictions: np.ndarray, row_count: int) -> pd.DataFrame:
    scored = frame[["row_id", "chunk_id"]].copy()
    scored["prediction"] = predictions.reshape(-1)
    grouped = scored.groupby("row_id", sort=True)["prediction"]
    output = pd.DataFrame(
        {
            "bert_mean": grouped.mean(),
            "bert_median": grouped.median(),
            "bert_first": grouped.first(),
            "bert_last": grouped.last(),
            "bert_min": grouped.min(),
            "bert_max": grouped.max(),
            "bert_chunk_count": grouped.size(),
        }
    ).reindex(range(row_count))
    if output.isna().any().any():
        raise RuntimeError("Missing test BERT prediction")
    return output.reset_index(drop=True)


set_seed(RANDOM_STATE)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
train = prepare(pd.read_csv("train.csv"))
test = prepare(pd.read_csv("test.csv"))
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

print("Building train chunks", flush=True)
train_chunks = build_chunks(train, tokenizer, include_labels=True)
print("Building test chunks", flush=True)
test_chunks = build_chunks(test, tokenizer, include_labels=False)
print("Chunks:", len(train_chunks), len(test_chunks), flush=True)

train_dataset = ChunkDataset(train_chunks, tokenizer, labels=True)
test_dataset = ChunkDataset(test_chunks, tokenizer, labels=False)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    problem_type="regression",
    local_files_only=True,
)
model.gradient_checkpointing_enable()

steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRADIENT_ACCUMULATION))
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=int(steps_per_epoch * EPOCHS * 0.1),
    eval_strategy="no",
    save_strategy="no",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    tf32=torch.cuda.is_available(),
    report_to=[],
    seed=RANDOM_STATE,
    dataloader_num_workers=0,
    save_safetensors=False,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()

# Predict before saving so a Windows rename/antivirus lock cannot lose the run.
chunk_predictions = trainer.predict(test_dataset).predictions.reshape(-1).astype("float32")
test_output = aggregate(test_chunks, chunk_predictions, len(test))
test_output.to_csv(TEST_PREDICTIONS, index=False, encoding="utf-8-sig")
print(test_output.describe(), flush=True)
print("Saved predictions:", TEST_PREDICTIONS, flush=True)

try:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved model:", OUTPUT_DIR, flush=True)
except Exception as error:
    print("Model save skipped after successful predictions:", repr(error), flush=True)

del trainer, model, train_dataset, test_dataset
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
