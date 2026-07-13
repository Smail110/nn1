"""Fine-tune ruBERT-large as a new, independent salary ensemble member."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET = "log_salary_from"
BASE_MODEL = Path("rubert_large_base")
LOCAL_DIR = Path("rubert_large_salary_local_081")
FULL_DIR = Path("rubert_large_salary_full_081")
LOCAL_PREDICTIONS = Path("local_valid_rubert_large_081.csv")
TEST_PREDICTIONS = Path("test_rubert_large_081.csv")
REPORT = Path("rubert_large_081_results.json")
SEED = 2026
MAX_LENGTH = 256
TEXT_SEPARATOR = "[SEP]"
BATCH_SIZE = 4
ACCUMULATION = 4
LOCAL_EPOCHS = 3
LEARNING_RATE = 1e-5


def make_text(frame: pd.DataFrame) -> pd.Series:
    frame = frame.copy()
    for column in ["title", "skills", "location", "company", "description"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    experience = pd.to_numeric(
        frame["experience_from"], errors="coerce"
    ).fillna(0).astype(str)
    # Put all short structured fields before the long description so truncation
    # cannot discard location, employer and experience.
    return (
        "Название: " + frame["title"] + " " + TEXT_SEPARATOR + " "
        + "Навыки: " + frame["skills"] + " " + TEXT_SEPARATOR + " "
        + "Регион: " + frame["location"] + " " + TEXT_SEPARATOR + " "
        + "Компания: " + frame["company"] + " " + TEXT_SEPARATOR + " "
        + "Опыт от: " + experience + " " + TEXT_SEPARATOR + " "
        + "Описание: " + frame["description"]
    )


class SalaryDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {name: values[index] for name, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def metrics(output):
    predictions, labels = output
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"r2": float(r2_score(labels, predictions))}


def model_and_tokenizer():
    global TEXT_SEPARATOR
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    TEXT_SEPARATOR = tokenizer.sep_token or "[SEP]"
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        problem_type="regression",
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def local_stage():
    set_seed(SEED)
    train = pd.read_csv("train.csv")
    train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
    train_part, valid = train_test_split(
        train,
        test_size=0.2,
        random_state=42,
        stratify=train["salary_bin"],
    )
    train_part = train_part.drop(columns="salary_bin").reset_index(drop=True)
    valid = valid.drop(columns="salary_bin").reset_index(drop=True)

    mean = float(train_part[TARGET].mean())
    std = float(train_part[TARGET].std())
    train_labels = (
        (train_part[TARGET].to_numpy(dtype="float32") - mean) / std
    ).astype("float32")
    valid_labels = (
        (valid[TARGET].to_numpy(dtype="float32") - mean) / std
    ).astype("float32")

    model, tokenizer = model_and_tokenizer()
    train_dataset = SalaryDataset(make_text(train_part), tokenizer, train_labels)
    valid_dataset = SalaryDataset(make_text(valid), tokenizer, valid_labels)
    steps_per_epoch = math.ceil(
        len(train_dataset) / (BATCH_SIZE * ACCUMULATION)
    )
    args = TrainingArguments(
        output_dir=str(LOCAL_DIR),
        overwrite_output_dir=True,
        num_train_epochs=LOCAL_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=int(steps_per_epoch * LOCAL_EPOCHS * 0.08),
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="r2",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    standardized = trainer.predict(valid_dataset).predictions.reshape(-1)
    prediction = standardized * std + mean
    standalone_score = float(r2_score(valid[TARGET], prediction))

    current = pd.read_csv("local_valid_final_boost_081.csv")
    current_prediction = current["final_boost_prediction"].to_numpy(dtype="float64")
    if not np.allclose(current[TARGET], valid[TARGET], atol=1e-7):
        raise RuntimeError("Validation row order mismatch")

    best = (float(r2_score(valid[TARGET], current_prediction)), 0.0, 1.0, 0.0)
    for weight in np.linspace(0.0, 0.5, 101):
        blended = (1.0 - weight) * current_prediction + weight * prediction
        slope, intercept = np.polyfit(blended, valid[TARGET], deg=1)
        calibrated = slope * blended + intercept
        score = float(r2_score(valid[TARGET], calibrated))
        if score > best[0]:
            best = (score, float(weight), float(slope), float(intercept))

    output = valid[["title", "location", "company", TARGET]].copy()
    output["rubert_large_prediction"] = prediction
    output["current_prediction"] = current_prediction
    output.to_csv(LOCAL_PREDICTIONS, index=False, encoding="utf-8-sig")

    best_checkpoint = None
    best_epoch = LOCAL_EPOCHS

    report = {
        "standalone_local_r2": standalone_score,
        "ensemble_local_r2": best[0],
        "ensemble_weight": best[1],
        "calibration": [best[2], best[3]],
        "best_epoch": int(best_epoch),
        "best_checkpoint": best_checkpoint,
        "target_mean": mean,
        "target_std": std,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def full_stage():
    set_seed(SEED)
    report = json.loads(REPORT.read_text("utf-8"))
    epochs = int(report["best_epoch"])
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    mean = float(train[TARGET].mean())
    std = float(train[TARGET].std())
    labels = ((train[TARGET].to_numpy(dtype="float32") - mean) / std).astype(
        "float32"
    )

    model, tokenizer = model_and_tokenizer()
    train_dataset = SalaryDataset(make_text(train), tokenizer, labels)
    test_dataset = SalaryDataset(make_text(test), tokenizer)
    steps_per_epoch = math.ceil(
        len(train_dataset) / (BATCH_SIZE * ACCUMULATION)
    )
    args = TrainingArguments(
        output_dir=str(FULL_DIR),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=int(steps_per_epoch * epochs * 0.08),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    standardized = trainer.predict(test_dataset).predictions.reshape(-1)
    prediction = standardized * std + mean
    pd.DataFrame({"rubert_large_prediction": prediction}).to_csv(
        TEST_PREDICTIONS, index=False, encoding="utf-8-sig"
    )
    print(pd.Series(prediction).describe(), flush=True)
    print("Saved:", TEST_PREDICTIONS, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["local", "full", "all"], default="local")
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
parser.add_argument("--epochs", type=int, default=LOCAL_EPOCHS)
parser.add_argument("--tag", default="")
parser.add_argument("--base-model", default=str(BASE_MODEL))
arguments = parser.parse_args()
SEED = arguments.seed
MAX_LENGTH = arguments.max_length
LOCAL_EPOCHS = arguments.epochs
BASE_MODEL = Path(arguments.base_model)
if arguments.tag:
    suffix = "_" + arguments.tag
    LOCAL_DIR = Path(f"rubert_large_salary_local_081{suffix}")
    FULL_DIR = Path(f"rubert_large_salary_full_081{suffix}")
    LOCAL_PREDICTIONS = Path(f"local_valid_rubert_large_081{suffix}.csv")
    TEST_PREDICTIONS = Path(f"test_rubert_large_081{suffix}.csv")
    REPORT = Path(f"rubert_large_081_results{suffix}.json")
if arguments.stage in ("local", "all"):
    local_stage()
if arguments.stage in ("full", "all"):
    full_stage()
