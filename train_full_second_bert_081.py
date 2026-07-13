"""Train the full-data counterpart of the honest single-sequence BERT."""

from pathlib import Path
import math
import os

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
BASE_MODEL = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--DeepPavlov--rubert-base-cased" / "snapshots"
    / "4036cab694767a299f2b9e6492909664d9414229"
)
OUTPUT_DIR = Path("rubert_salary_full_second_081")
OUTPUT_PREDICTIONS = Path("test_full_second_bert_081.csv")


def make_text(frame: pd.DataFrame) -> pd.Series:
    for column in ["title", "skills", "description", "location", "company"]:
        frame[column] = frame[column].fillna("").astype(str)
    return (
        "Название: " + frame["title"] + " [SEP] "
        + "Навыки: " + frame["skills"] + " [SEP] "
        + "Описание: " + frame["description"] + " [SEP] "
        + "Регион: " + frame["location"] + " [SEP] "
        + "Компания: " + frame["company"] + " [SEP] "
        + "Опыт от: " + frame["experience_from"].astype(str)
    )


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None):
        self.encodings = tokenizer(
            texts.tolist(), truncation=True, max_length=512, padding=False
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


set_seed(42)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_text = make_text(train)
test_text = make_text(test)
target_mean = float(train[TARGET].mean())
target_std = float(train[TARGET].std())
labels = ((train[TARGET].to_numpy(dtype="float32") - target_mean) / target_std).astype("float32")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
train_dataset = TextDataset(train_text, tokenizer, labels)
test_dataset = TextDataset(test_text, tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    problem_type="regression",
    local_files_only=True,
)
model.gradient_checkpointing_enable()

batch_size = 8
accumulation = 2
epochs = 4
steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * accumulation))
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=accumulation,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=int(steps_per_epoch * epochs * 0.1),
    eval_strategy="no",
    save_strategy="no",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    tf32=torch.cuda.is_available(),
    report_to=[],
    seed=42,
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

standardized = trainer.predict(test_dataset).predictions.reshape(-1).astype("float32")
prediction = standardized * target_std + target_mean
pd.DataFrame(
    {
        "second_bert_standardized": standardized,
        "second_bert_prediction": prediction,
    }
).to_csv(OUTPUT_PREDICTIONS, index=False, encoding="utf-8-sig")
print(pd.Series(prediction).describe(), flush=True)
print("Saved:", OUTPUT_PREDICTIONS, flush=True)

try:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
except Exception as error:
    print("Model save skipped after predictions:", repr(error), flush=True)
