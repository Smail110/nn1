"""Train/evaluate a leakage-free chunked ruBERT and blend direct predictions."""

from __future__ import annotations

import gc
import math
import os
import shutil
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
    Trainer,
    TrainingArguments,
    set_seed,
)


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET = "log_salary_from"
RANDOM_STATE = 42
VALID_SIZE = 0.2

BASE_MODEL = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--DeepPavlov--rubert-base-cased"
    / "snapshots"
    / "4036cab694767a299f2b9e6492909664d9414229"
)
LOCAL_MODEL_DIR = Path("rubert_salary_honest_20pct_079")
FULL_MODEL_DIR = Path("rubert_salary_chunked_full_train")

MAX_LENGTH = 512
PREFIX_MAX_TOKENS = 160
DESC_CHUNK_TOKENS = 320
DESC_CHUNK_OVERLAP = 80
MAX_CHUNKS_PER_ROW = 6
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
EPOCHS = 4
LEARNING_RATE = 1e-5
REUSE_LOCAL_MODEL = True

LOCAL_BERT_PREDICTIONS = Path("local_valid_honest_bert_079.csv")
TEST_BERT_PREDICTIONS = Path("test_honest_full_bert_079.csv")
INPUT_VALID_BLEND = Path("local_valid_079_tfidf_te_predictions.csv")
INPUT_TEST_BLEND = Path("submission_079_tfidf_te_blend.csv")
OUTPUT_SUBMISSION = Path("submission_079_honest_bert_blend.csv")
OUTPUT_SUBMISSION_UNCALIBRATED = Path("submission_079_honest_bert_blend_uncalibrated.csv")


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
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


def split_description(tokenizer, description: str) -> list[list[int]]:
    token_ids = tokenizer(str(description), add_special_tokens=False)["input_ids"]
    if not token_ids:
        return [[]]
    chunks = []
    start = 0
    while start < len(token_ids) and len(chunks) < MAX_CHUNKS_PER_ROW:
        chunks.append(token_ids[start : start + DESC_CHUNK_TOKENS])
        if start + DESC_CHUNK_TOKENS >= len(token_ids):
            break
        start += DESC_CHUNK_TOKENS - DESC_CHUNK_OVERLAP
    return chunks


def build_chunks(df: pd.DataFrame, tokenizer, include_labels: bool) -> pd.DataFrame:
    rows = []
    for row_id, row in df.reset_index(drop=True).iterrows():
        prefix_ids = tokenizer(
            make_prefix(row),
            add_special_tokens=False,
            truncation=True,
            max_length=PREFIX_MAX_TOKENS,
        )["input_ids"]
        for chunk_id, desc_ids in enumerate(split_description(tokenizer, row["description"])):
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
            rows.append(item)
    return pd.DataFrame(rows)


class SalaryChunks(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, include_labels: bool):
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = (
            frame["label"].to_numpy(dtype="float32") if include_labels else None
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def aggregate_chunk_predictions(
    chunk_frame: pd.DataFrame, predictions: np.ndarray, row_count: int
) -> pd.DataFrame:
    scored = chunk_frame[["row_id", "chunk_id"]].copy()
    scored["prediction"] = np.asarray(predictions).reshape(-1)
    grouped = scored.groupby("row_id", sort=True)["prediction"]
    result = pd.DataFrame(
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
    if result.isna().any().any():
        raise RuntimeError("Missing aggregated BERT predictions")
    return result.reset_index(drop=True)


def predict_chunks(model_dir: Path, chunk_frame: pd.DataFrame) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    dataset = SalaryChunks(chunk_frame, tokenizer, include_labels=False)
    inference_output_dir = Path("bert_inference_tmp") / model_dir.name
    args = TrainingArguments(
        output_dir=str(inference_output_dir),
        per_device_eval_batch_size=BATCH_SIZE * 2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions = trainer.predict(dataset).predictions.reshape(-1).astype("float32")
    del trainer, dataset, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


set_seed(RANDOM_STATE)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

train = prepare_frame(pd.read_csv("train.csv"))
test = prepare_frame(pd.read_csv("test.csv"))
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, valid_part = train_test_split(
    train,
    test_size=VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=train["salary_bin"],
)
train_part = train_part.drop(columns="salary_bin").reset_index(drop=True)
valid_part = valid_part.drop(columns="salary_bin").reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
train_chunks = build_chunks(train_part, tokenizer, include_labels=True)
valid_chunks = build_chunks(valid_part, tokenizer, include_labels=True)
print("Chunks train/valid:", len(train_chunks), len(valid_chunks))

if REUSE_LOCAL_MODEL and (LOCAL_MODEL_DIR / "model.safetensors").exists():
    print("Reusing trained honest local model:", LOCAL_MODEL_DIR)
    del train_chunks
else:
    if LOCAL_MODEL_DIR.exists():
        shutil.rmtree(LOCAL_MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        problem_type="regression",
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()

    train_dataset = SalaryChunks(train_chunks, tokenizer, include_labels=True)
    steps_per_epoch = math.ceil(
        len(train_dataset) / (BATCH_SIZE * GRADIENT_ACCUMULATION)
    )
    training_args = TrainingArguments(
        output_dir=str(LOCAL_MODEL_DIR),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
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
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    del trainer, train_dataset, model, train_chunks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

valid_chunk_predictions = predict_chunks(LOCAL_MODEL_DIR, valid_chunks)
valid_bert = aggregate_chunk_predictions(
    valid_chunks, valid_chunk_predictions, len(valid_part)
)
y_valid = valid_part[TARGET].to_numpy(dtype="float32")
for column in ["bert_mean", "bert_median", "bert_first", "bert_last", "bert_min", "bert_max"]:
    print(column, f"R2={r2_score(y_valid, valid_bert[column]):.6f}")

saved_valid = pd.read_csv(INPUT_VALID_BLEND)
if not np.allclose(saved_valid[TARGET], y_valid, atol=1e-6):
    raise RuntimeError("Validation order mismatch")
base_valid = saved_valid["blend_pred_calibrated"].to_numpy(dtype="float32")

best = (-np.inf, None, None)
for column in ["bert_mean", "bert_median", "bert_first", "bert_last", "bert_min", "bert_max"]:
    bert_prediction = valid_bert[column].to_numpy(dtype="float32")
    for weight in np.linspace(0.0, 0.6, 61):
        prediction = (1.0 - weight) * base_valid + weight * bert_prediction
        score = r2_score(y_valid, prediction)
        if score > best[0]:
            best = (float(score), column, float(weight))
print("Best honest BERT blend:", best)

valid_output = valid_part[["title", "location", "company", "experience_from", TARGET]].copy()
valid_output = pd.concat([valid_output, valid_bert], axis=1)
valid_output["input_blend"] = base_valid
valid_output["final_blend"] = (
    (1.0 - best[2]) * base_valid + best[2] * valid_bert[best[1]].to_numpy()
)
valid_output.to_csv(LOCAL_BERT_PREDICTIONS, index=False, encoding="utf-8-sig")

full_tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_DIR, local_files_only=True)
test_chunks = build_chunks(test, full_tokenizer, include_labels=False)
del full_tokenizer
test_chunk_predictions = predict_chunks(FULL_MODEL_DIR, test_chunks)
test_bert = aggregate_chunk_predictions(test_chunks, test_chunk_predictions, len(test))
test_bert.to_csv(TEST_BERT_PREDICTIONS, index=False, encoding="utf-8-sig")

submission = pd.read_csv(INPUT_TEST_BLEND)
prediction_column = "prediction" if "prediction" in submission else submission.columns[-1]
input_test = submission[prediction_column].to_numpy(dtype="float32")
final_test_uncalibrated = (
    (1.0 - best[2]) * input_test
    + best[2] * test_bert[best[1]].to_numpy(dtype="float32")
)

submission[prediction_column] = final_test_uncalibrated
submission.to_csv(OUTPUT_SUBMISSION_UNCALIBRATED, index=False, encoding="utf-8-sig")

final_valid_uncalibrated = valid_output["final_blend"].to_numpy(dtype="float32")
calibration = np.polyfit(final_valid_uncalibrated, y_valid, deg=1)
final_test = calibration[0] * final_test_uncalibrated + calibration[1]
submission[prediction_column] = final_test
submission.to_csv(OUTPUT_SUBMISSION, index=False, encoding="utf-8-sig")

print("Final local R2:", f"{best[0]:.6f}")
print("Calibration:", calibration.tolist())
print("Saved:", OUTPUT_SUBMISSION)
