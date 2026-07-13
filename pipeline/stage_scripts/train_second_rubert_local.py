"""РЎС‚Р°РґРёСЏ 10: РѕР±СѓС‡РµРЅРёРµ local-valid BERT РјРѕРґРµР»Рё РґР»СЏ С‡РµСЃС‚РЅРѕРіРѕ OOF.

Р—Р°С‡РµРј СЌС‚Рѕ РЅСѓР¶РЅРѕ:
- С„РёРЅР°Р»СЊРЅС‹Р№ stack РёСЃРїРѕР»СЊР·СѓРµС‚ `local_valid_second_bert_081.csv`;
- СЌС‚РѕС‚ CSV РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РїРѕР»СѓС‡РµРЅ С‡РµСЃС‚РЅРѕ: РјРѕРґРµР»СЊ СѓС‡РёС‚СЃСЏ С‚РѕР»СЊРєРѕ РЅР° train-part,
  Р° РїСЂРµРґСЃРєР°Р·С‹РІР°РµС‚ holdout-valid;
- РїРѕСЌС‚РѕРјСѓ РјС‹ РЅРµ СЃС‡РёС‚Р°РµРј `rubert_salary_local_valid_only` РІРЅРµС€РЅРёРј Р°СЂС‚РµС„Р°РєС‚РѕРј,
  Р° РїРµСЂРµРѕР±СѓС‡Р°РµРј РµРіРѕ РёР· `train.csv`.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
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

from config import DEEPPAVLOV_RUBERT_BASE, ROOT


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET = "log_salary_from"
OUTPUT_DIR = ROOT / "rubert_salary_local_valid_only"


def make_text(frame: pd.DataFrame) -> pd.Series:
    """РЎРѕР±РёСЂР°РµС‚ РѕРґРёРЅ С‚РµРєСЃС‚ РІР°РєР°РЅСЃРёРё РёР· РІСЃРµС… РїРѕР»РµР№, РєР°Рє РІ evaluate_second_rubert.py."""

    frame = frame.copy()
    for column in ["title", "skills", "description", "location", "company"]:
        frame[column] = frame[column].fillna("").astype(str)
    return (
        "РќР°Р·РІР°РЅРёРµ: " + frame["title"] + " [SEP] "
        + "РќР°РІС‹РєРё: " + frame["skills"] + " [SEP] "
        + "РћРїРёСЃР°РЅРёРµ: " + frame["description"] + " [SEP] "
        + "Р РµРіРёРѕРЅ: " + frame["location"] + " [SEP] "
        + "РљРѕРјРїР°РЅРёСЏ: " + frame["company"] + " [SEP] "
        + "РћРїС‹С‚ РѕС‚: " + frame["experience_from"].astype(str)
    )


class TextDataset(Dataset):
    """РњРёРЅРёРјР°Р»СЊРЅС‹Р№ Dataset РґР»СЏ Hugging Face Trainer."""

    def __init__(self, texts: pd.Series, tokenizer: AutoTokenizer, labels: pd.Series | None = None):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=512,
            padding=False,
        )
        self.labels = None if labels is None else labels.to_numpy(dtype="float32")

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: values[index] for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def main() -> None:
    if not DEEPPAVLOV_RUBERT_BASE.exists():
        raise SystemExit(
            "РќРµ РЅР°Р№РґРµРЅ DeepPavlov/rubert-base-cased РІ Hugging Face cache: "
            f"{DEEPPAVLOV_RUBERT_BASE}"
        )

    set_seed(42)
    train = pd.read_csv(ROOT / "train.csv")
    train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
    train_part, _ = train_test_split(
        train,
        test_size=0.2,
        random_state=42,
        stratify=train["salary_bin"],
    )
    train_part = train_part.drop(columns="salary_bin").reset_index(drop=True)

    print("[local-second-bert] train rows:", len(train_part), flush=True)
    print("[local-second-bert] base:", DEEPPAVLOV_RUBERT_BASE, flush=True)
    print("[local-second-bert] output:", OUTPUT_DIR, flush=True)

    if OUTPUT_DIR.exists():
        # Р­С‚Рѕ generated-output С‚РµРєСѓС‰РµР№ СЃС‚Р°РґРёРё. РЈРґР°Р»СЏРµРј С‚РѕР»СЊРєРѕ РїРѕСЃР»Рµ РїСЂРѕРІРµСЂРєРё,
        # С‡С‚РѕР±С‹ РІ РїР°РїРєРµ РЅРµ РѕСЃС‚Р°Р»РёСЃСЊ СЃС‚Р°СЂС‹Рµ model.safetensors/pytorch_model.bin.
        if OUTPUT_DIR.resolve().parent != ROOT.resolve():
            raise SystemExit(f"РќРµР±РµР·РѕРїР°СЃРЅС‹Р№ output path: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(DEEPPAVLOV_RUBERT_BASE, local_files_only=True)
    dataset = TextDataset(make_text(train_part), tokenizer, train_part[TARGET])

    model = AutoModelForSequenceClassification.from_pretrained(
        DEEPPAVLOV_RUBERT_BASE,
        num_labels=1,
        problem_type="regression",
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()

    batch_size = 8
    accumulation = 2
    epochs = 4
    steps_per_epoch = math.ceil(len(dataset) / (batch_size * accumulation))

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
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[local-second-bert] saved:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()

