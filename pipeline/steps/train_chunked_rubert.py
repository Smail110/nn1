from autogluon.tabular import TabularPredictor

import pandas as pd
import numpy as np
import torch
import os
import shutil
import math
import re

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


# ============================================================
# 0. НАСТРОЙКИ
# ============================================================

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
FALLBACK_SAMPLE_SUBMISSION_PATH = "sample_submition.csv"

SUBMISSION_PATH = "submission.csv"
SUBMISSION_LOG_PATH = "submission_log.csv"

LOCAL_RUBERT_CACHE = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "models--DeepPavlov--rubert-base-cased",
    "snapshots",
    "4036cab694767a299f2b9e6492909664d9414229",
)
BASE_RUBERT_MODEL = os.environ.get(
    "BASE_RUBERT_MODEL",
    LOCAL_RUBERT_CACHE if os.path.exists(LOCAL_RUBERT_CACHE) else "DeepPavlov/rubert-base-cased",
)

TARGET_COLUMN = "log_salary_from"
SALARY_COLUMN = "salary_from"

TEXT_COLUMNS = ["title", "location", "company", "skills", "description"]

MAX_LENGTH = 512

PREFIX_MAX_TOKENS = 160
DESC_CHUNK_TOKENS = 320
DESC_CHUNK_OVERLAP = 80
MAX_CHUNKS_PER_ROW = 6

BERT_BATCH_SIZE = 8
BERT_EPOCHS = 4
BERT_LR = 1e-5

AUTOGLOUON_TIME_LIMIT = 900

RANDOM_STATE = 42

RUBERT_OUTPUT_DIR = "rubert_salary_chunked_full_train"
AUTOGLOUON_PATH = "autogluon_salary_kaggle_full_train_leak_features"

USE_RAW_TEXT_FEATURES_IN_AUTOGLUON = True

AGGREGATIONS = ["mean", "max", "first", "last"]

REUSE_SAVED_VACANCY_EMBEDDINGS = True
TRAIN_VACANCY_EMBEDDINGS_PATH = "kaggle_train_chunked_vacancy_embeddings.npy"
TEST_VACANCY_EMBEDDINGS_PATH = "kaggle_test_chunked_vacancy_embeddings.npy"

USE_SALARY_TEXT_LEAK_FEATURES = True
USE_SALARY_HINT_POSTPROCESS = False
SALARY_HINT_POSTPROCESS_WEIGHT = 0.025
SALARY_HINT_POSTPROCESS_MIN = 50.0

SALARY_TEXT_LEAK_FEATURE_COLUMNS = [
    "salary_hint_count",
    "salary_hint_min",
    "salary_hint_max",
    "salary_hint_mean",
    "salary_hint_first",
    "salary_hint_last",
    "salary_hint_spread",
    "salary_hint_first_log",
    "salary_hint_min_log",
    "salary_hint_max_log",
    "salary_hint_mean_log",
    "salary_hint_last_log",
    "salary_hint_has",
    "salary_hint_tight",
]

# В финальном Kaggle-скрипте пока без весов и без калибровки.
# Сначала смотрим чистый вариант: raw признаки + BERT embeddings.
USE_SAMPLE_WEIGHTS = False
SAMPLE_WEIGHT_COLUMN = "sample_weight"


# ============================================================
# 1. GPU CHECK
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("DEVICE:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
else:
    print("GPU не найден, обучение будет на CPU.")

print("=" * 80)


# ============================================================
# 2. ЗАГРУЗКА TRAIN / TEST / SAMPLE SUBMISSION
# ============================================================

if not os.path.exists(SAMPLE_SUBMISSION_PATH):
    if os.path.exists(FALLBACK_SAMPLE_SUBMISSION_PATH):
        print(
            f"{SAMPLE_SUBMISSION_PATH} не найден, использую "
            f"{FALLBACK_SAMPLE_SUBMISSION_PATH}"
        )
        SAMPLE_SUBMISSION_PATH = FALLBACK_SAMPLE_SUBMISSION_PATH
    else:
        raise FileNotFoundError(
            f"Не найден ни {SAMPLE_SUBMISSION_PATH}, ни "
            f"{FALLBACK_SAMPLE_SUBMISSION_PATH}"
        )

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

required_train_columns = TEXT_COLUMNS + ["experience_from", TARGET_COLUMN, SALARY_COLUMN]
required_test_columns = TEXT_COLUMNS + ["experience_from"]

for col in required_train_columns:
    if col not in train_df.columns:
        raise ValueError(f"В train.csv нет обязательной колонки: {col}")

for col in required_test_columns:
    if col not in test_df.columns:
        raise ValueError(f"В test.csv нет обязательной колонки: {col}")

train_df = train_df.dropna(subset=[TARGET_COLUMN, SALARY_COLUMN]).copy()
train_df = train_df[train_df[SALARY_COLUMN] > 0].copy()

for df in [train_df, test_df]:
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["experience_from"] = df["experience_from"].fillna(0).astype(float)

train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype("float32")
train_df[SALARY_COLUMN] = train_df[SALARY_COLUMN].astype("float32")


def add_full_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["full_text"] = (
        "Название: " + df["title"] + " [SEP] " +
        "Навыки: " + df["skills"] + " [SEP] " +
        "Описание: " + df["description"] + " [SEP] " +
        "Регион: " + df["location"] + " [SEP] " +
        "Компания: " + df["company"] + " [SEP] " +
        "Опыт от: " + df["experience_from"].astype(str)
    )

    return df


SALARY_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{5,7}|\d{2,4}(?:[,.]\d+)?)(?!\d)"
)

SALARY_KEYWORD_PATTERN = re.compile(
    r"(зарплат|заработн|окладн|оклад|доход|зп)",
    flags=re.IGNORECASE,
)

RUBLE_PATTERN = re.compile(r"(руб|₽|р\b)", flags=re.IGNORECASE)
THOUSAND_PATTERN = re.compile(r"(тыс|k\b)", flags=re.IGNORECASE)
USD_PATTERN = re.compile(r"(\$|usd|долл)", flags=re.IGNORECASE)

BAD_AMOUNT_UNIT_PATTERN = re.compile(
    r"(%|/\s*ч|в\s*час|час|/\s*м|м2|м²|"
    r"за\s+одн|за\s+шт|за\s+план|"
    r"дн(?:ей|я|ь)|лет|год|месяц|"
    r"сотрудн|метр|офис|пеш)",
    flags=re.IGNORECASE,
)


def parse_salary_number(value: str):
    raw = str(value).strip().replace("\u00a0", " ")

    grouped = bool(
        re.search(r"\d[\s\u00a0.]\d{3}\b", raw) or
        re.search(r"\d,\d{3}\b", raw)
    )

    cleaned = re.sub(r"(?<=\d)[\s\u00a0](?=\d)", "", raw)

    if grouped:
        cleaned = cleaned.replace(".", "").replace(",", "")
    else:
        cleaned = cleaned.replace(",", ".")

    cleaned = re.sub(r"[^0-9.]", "", cleaned)

    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")

    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_salary_amount_to_thousands(amount, context: str):
    if amount is None or amount > 1_200_000:
        return None

    context = str(context)

    if USD_PATTERN.search(context):
        # Грубая конвертация валютных вилок: $1000 -> около 90 тыс. руб.
        amount = amount * 0.09
    elif amount >= 5000:
        amount = amount / 1000.0
    elif THOUSAND_PATTERN.search(context):
        amount = amount
    elif RUBLE_PATTERN.search(context):
        # 500 руб. рядом с зарплатой чаще надбавка/часовая ставка, не 500 тыс.
        return None
    elif amount < 20:
        return None

    if 5 <= amount <= 1200:
        return float(amount)

    return None


def extract_salary_hints_from_text(text: str):
    text = re.sub(
        r"\s+",
        " ",
        str(text).lower().replace("\u00a0", " ")
    )

    hints = []

    for keyword_match in SALARY_KEYWORD_PATTERN.finditer(text):
        segment = text[
            keyword_match.start():
            min(len(text), keyword_match.start() + 170)
        ]

        for number_match in SALARY_NUMBER_PATTERN.finditer(segment):
            local_context = segment[
                max(0, number_match.start() - 20):
                min(len(segment), number_match.end() + 35)
            ]

            raw_amount = parse_salary_number(number_match.group())

            is_large_monthly_ruble_amount = (
                raw_amount is not None and
                raw_amount >= 5000 and
                (
                    RUBLE_PATTERN.search(local_context) or
                    THOUSAND_PATTERN.search(local_context)
                )
            )

            if (
                BAD_AMOUNT_UNIT_PATTERN.search(local_context) and
                not is_large_monthly_ruble_amount
            ):
                continue

            amount = normalize_salary_amount_to_thousands(
                raw_amount,
                local_context
            )

            if amount is not None:
                hints.append(amount)

    unique_hints = []

    for amount in hints:
        if not any(abs(amount - known) < 0.01 for known in unique_hints):
            unique_hints.append(amount)

    return unique_hints


def build_salary_search_text(df: pd.DataFrame) -> pd.Series:
    return df[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)


def add_salary_text_leak_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    search_text = build_salary_search_text(df)

    rows = []

    for text in search_text:
        hints = extract_salary_hints_from_text(text)

        if hints:
            values = np.asarray(hints, dtype="float32")

            item = {
                "salary_hint_count": float(len(hints)),
                "salary_hint_min": float(values.min()),
                "salary_hint_max": float(values.max()),
                "salary_hint_mean": float(values.mean()),
                "salary_hint_first": float(hints[0]),
                "salary_hint_last": float(hints[-1]),
                "salary_hint_spread": float(values.max() - values.min()),
                "salary_hint_has": 1.0,
                "salary_hint_tight": float(values.max() - values.min() <= 50),
            }
        else:
            item = {
                "salary_hint_count": 0.0,
                "salary_hint_min": np.nan,
                "salary_hint_max": np.nan,
                "salary_hint_mean": np.nan,
                "salary_hint_first": np.nan,
                "salary_hint_last": np.nan,
                "salary_hint_spread": np.nan,
                "salary_hint_has": 0.0,
                "salary_hint_tight": 0.0,
            }

        for col in [
            "salary_hint_first",
            "salary_hint_min",
            "salary_hint_max",
            "salary_hint_mean",
            "salary_hint_last",
        ]:
            value = item[col]
            item[f"{col}_log"] = (
                float(np.log(value))
                if pd.notna(value) and value > 0
                else np.nan
            )

        rows.append(item)

    features = pd.DataFrame(rows, index=df.index)

    return pd.concat([df, features], axis=1)


def salary_from_log(log_values):
    log_values = np.asarray(log_values, dtype="float32")
    log_values = np.clip(log_values, 0, 20)

    return np.exp(log_values)


def apply_salary_hint_postprocess(
    pred_log_values,
    df: pd.DataFrame,
    weight: float = 0.10,
):
    pred_log_values = np.asarray(pred_log_values, dtype="float32").copy()

    if "salary_hint_first_log" not in df.columns:
        return pred_log_values, 0

    hint_log = df["salary_hint_first_log"].to_numpy(dtype="float32")
    hint_salary = df["salary_hint_first"].to_numpy(dtype="float32")
    mask = (
        np.isfinite(hint_log) &
        np.isfinite(hint_salary) &
        (hint_salary >= SALARY_HINT_POSTPROCESS_MIN)
    )

    pred_log_values[mask] = (
        (1.0 - weight) * pred_log_values[mask] +
        weight * hint_log[mask]
    )

    return pred_log_values, int(mask.sum())


train_df = add_full_text(train_df).reset_index(drop=True)
test_df = add_full_text(test_df).reset_index(drop=True)

if USE_SALARY_TEXT_LEAK_FEATURES:
    train_df = add_salary_text_leak_features(train_df)
    test_df = add_salary_text_leak_features(test_df)

    train_hint_mask = train_df["salary_hint_first"].notna()
    test_hint_mask = test_df["salary_hint_first"].notna()

    print("Salary text hints train:", int(train_hint_mask.sum()), "/", len(train_df))
    print("Salary text hints test:", int(test_hint_mask.sum()), "/", len(test_df))

    if train_hint_mask.any():
        hint_abs_error = np.abs(
            train_df.loc[train_hint_mask, "salary_hint_first"].values -
            train_df.loc[train_hint_mask, SALARY_COLUMN].values
        )

        print(
            "Salary hint first MAE on train rows with hint:",
            float(hint_abs_error.mean())
        )
        print(
            "Salary hint first within 5k:",
            float((hint_abs_error <= 5).mean())
        )

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)
print("Sample submission columns:", sample_submission.columns.tolist())
print("=" * 80)


# ============================================================
# 3. TOKENIZER И CHUNKING ПО ТОКЕНАМ
# ============================================================

TOKENIZER_SOURCE = (
    RUBERT_OUTPUT_DIR
    if os.path.exists(RUBERT_OUTPUT_DIR)
    else BASE_RUBERT_MODEL
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_SOURCE,
    local_files_only=os.path.exists(TOKENIZER_SOURCE)
)


def make_prefix(row) -> str:
    return (
        "Название: " + str(row["title"]) + " [SEP] " +
        "Навыки: " + str(row["skills"]) + " [SEP] " +
        "Регион: " + str(row["location"]) + " [SEP] " +
        "Компания: " + str(row["company"]) + " [SEP] " +
        "Опыт от: " + str(row["experience_from"]) + " [SEP] " +
        "Описание: "
    )


def split_description_token_chunks(description: str):
    desc_ids = tokenizer(
        str(description),
        add_special_tokens=False
    )["input_ids"]

    if len(desc_ids) == 0:
        return [[]]

    step = max(1, DESC_CHUNK_TOKENS - DESC_CHUNK_OVERLAP)

    chunks = []

    for start in range(0, len(desc_ids), step):
        chunk_ids = desc_ids[start:start + DESC_CHUNK_TOKENS]
        chunks.append(chunk_ids)

        if len(chunks) >= MAX_CHUNKS_PER_ROW:
            break

        if start + DESC_CHUNK_TOKENS >= len(desc_ids):
            break

    return chunks


def make_row_chunks(row):
    prefix = make_prefix(row)

    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=True,
        max_length=PREFIX_MAX_TOKENS
    )["input_ids"]

    desc_chunks = split_description_token_chunks(row["description"])

    row_chunks = []

    for desc_ids in desc_chunks:
        input_ids = prefix_ids + desc_ids
        input_ids = input_ids[:MAX_LENGTH - 2]

        chunk_text = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        row_chunks.append(chunk_text)

    if len(row_chunks) == 0:
        row_chunks = [prefix]

    return row_chunks


def build_chunk_dataframe(rows_df: pd.DataFrame, include_label: bool) -> pd.DataFrame:
    chunk_rows = []

    for row_id, row in rows_df.iterrows():
        chunks = make_row_chunks(row)

        for chunk_id, chunk_text in enumerate(chunks):
            item = {
                "row_id": row_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text
            }

            if include_label:
                item[TARGET_COLUMN] = row[TARGET_COLUMN]

            chunk_rows.append(item)

    return pd.DataFrame(chunk_rows)


print("Создаю чанки для train...")
train_chunks = build_chunk_dataframe(train_df, include_label=True)

print("Создаю чанки для test...")
test_chunks = build_chunk_dataframe(test_df, include_label=False)

print("Train chunks:", train_chunks.shape)
print("Test chunks:", test_chunks.shape)

print("Статистика чанков на train-вакансию:")
print(train_chunks.groupby("row_id").size().describe())

print("Статистика чанков на test-вакансию:")
print(test_chunks.groupby("row_id").size().describe())

print("=" * 80)



def get_chunk_counts(chunk_df: pd.DataFrame, n_rows: int):
    return (
        chunk_df.groupby("row_id")
        .size()
        .reindex(range(n_rows), fill_value=0)
        .astype("float32")
        .values
    )


use_saved_vacancy_embeddings = (
    REUSE_SAVED_VACANCY_EMBEDDINGS and
    os.path.exists(TRAIN_VACANCY_EMBEDDINGS_PATH) and
    os.path.exists(TEST_VACANCY_EMBEDDINGS_PATH) and
    os.path.exists(RUBERT_OUTPUT_DIR)
)

if use_saved_vacancy_embeddings:
    print("Загружаю готовые vacancy embeddings:")
    print(TRAIN_VACANCY_EMBEDDINGS_PATH)
    print(TEST_VACANCY_EMBEDDINGS_PATH)

    X_train_final = np.load(TRAIN_VACANCY_EMBEDDINGS_PATH).astype("float32")
    X_test_final = np.load(TEST_VACANCY_EMBEDDINGS_PATH).astype("float32")

    if X_train_final.shape[0] != len(train_df):
        raise ValueError(
            f"Размер train embeddings ({X_train_final.shape[0]}) "
            f"не совпадает с train_df ({len(train_df)})"
        )

    if X_test_final.shape[0] != len(test_df):
        raise ValueError(
            f"Размер test embeddings ({X_test_final.shape[0]}) "
            f"не совпадает с test_df ({len(test_df)})"
        )

    train_chunk_counts = get_chunk_counts(train_chunks, len(train_df))
    test_chunk_counts = get_chunk_counts(test_chunks, len(test_df))

    print("Train final vacancy embeddings:", X_train_final.shape)
    print("Test final vacancy embeddings:", X_test_final.shape)
    print("=" * 80)
else:
    if (
        REUSE_SAVED_VACANCY_EMBEDDINGS
        and os.path.exists(TRAIN_VACANCY_EMBEDDINGS_PATH)
        and os.path.exists(TEST_VACANCY_EMBEDDINGS_PATH)
        and not os.path.exists(RUBERT_OUTPUT_DIR)
    ):
        print(
            "Готовые vacancy embeddings найдены, но папка fine-tuned ruBERT "
            f"{RUBERT_OUTPUT_DIR!r} отсутствует. Перезапускаю fine-tuning, "
            "потому что следующая стадия blend_chunked_rubert.py загружает "
            "именно эту модель.",
            flush=True,
        )

    # ============================================================
    # 4. LABEL ДЛЯ ДОБООБУЧЕНИЯ BERT
    # ============================================================
    
    # BERT дообучаем на log_salary_from, но label нормализуем.
    # Это не финальный ответ модели, а способ сделать embeddings зарплатно-ориентированными.
    
    bert_y_mean = train_df[TARGET_COLUMN].mean()
    bert_y_std = train_df[TARGET_COLUMN].std()
    
    if bert_y_std == 0:
        raise ValueError("Стандартное отклонение target равно 0, обучение невозможно.")
    
    train_chunks["bert_label"] = (
        train_chunks[TARGET_COLUMN] - bert_y_mean
    ) / bert_y_std
    
    print("BERT label mean:", bert_y_mean)
    print("BERT label std:", bert_y_std)
    print("=" * 80)
    
    
    # ============================================================
    # 5. DATASET ДЛЯ BERT
    # ============================================================
    
    class ChunkSalaryDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=MAX_LENGTH
            )
    
            self.labels = labels.astype("float32").values
    
        def __len__(self):
            return len(self.labels)
    
        def __getitem__(self, idx):
            item = {
                key: torch.tensor(val[idx])
                for key, val in self.encodings.items()
            }
    
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
    
            return item
    
    
    bert_train_dataset = ChunkSalaryDataset(
        train_chunks["chunk_text"].tolist(),
        train_chunks["bert_label"]
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    
    # ============================================================
    # 6. ДОБООБУЧЕНИЕ ruBERT НА ВСЁМ TRAIN
    # ============================================================
    
    if os.path.exists(RUBERT_OUTPUT_DIR):
        shutil.rmtree(RUBERT_OUTPUT_DIR)
    
    bert_regressor = AutoModelForSequenceClassification.from_pretrained(
        BASE_RUBERT_MODEL,
        num_labels=1,
        problem_type="regression",
        local_files_only=os.path.exists(BASE_RUBERT_MODEL),
    )
    
    bert_regressor.to(device)
    
    if hasattr(bert_regressor, "gradient_checkpointing_enable"):
        bert_regressor.gradient_checkpointing_enable()
    
    num_update_steps_per_epoch = math.ceil(
        len(bert_train_dataset) / (BERT_BATCH_SIZE * 2)
    )
    
    total_steps = num_update_steps_per_epoch * BERT_EPOCHS
    warmup_steps = int(total_steps * 0.1)
    
    training_args = TrainingArguments(
        output_dir=RUBERT_OUTPUT_DIR,
        overwrite_output_dir=True,
    
        num_train_epochs=BERT_EPOCHS,
    
        per_device_train_batch_size=BERT_BATCH_SIZE,
        gradient_accumulation_steps=2,
    
        learning_rate=BERT_LR,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
    
        eval_strategy="no",
        save_strategy="epoch",
    
        load_best_model_at_end=False,
    
        logging_steps=50,
        save_total_limit=1,
    
        fp16=torch.cuda.is_available(),
    
        report_to=[]
    )
    
    trainer = Trainer(
        model=bert_regressor,
        args=training_args,
        train_dataset=bert_train_dataset,
        data_collator=data_collator
    )
    
    print("\nНачинаю fine-tuning ruBERT на всём train...")
    trainer.train()
    
    trainer.save_model(RUBERT_OUTPUT_DIR)
    tokenizer.save_pretrained(RUBERT_OUTPUT_DIR)
    
    print("\nChunked ruBERT сохранён в:", RUBERT_OUTPUT_DIR)
    print("=" * 80)
    
    del bert_regressor
    del trainer
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    
    # ============================================================
    # 7. CHUNK EMBEDDINGS ИЗ ДОБУЧЕННОГО ruBERT
    # ============================================================
    
    tokenizer = AutoTokenizer.from_pretrained(RUBERT_OUTPUT_DIR)
    
    bert_encoder = AutoModel.from_pretrained(
        RUBERT_OUTPUT_DIR,
        output_hidden_states=False
    )
    
    bert_encoder.to(device)
    bert_encoder.eval()
    
    
    @torch.no_grad()
    def get_chunk_embeddings(texts, batch_size=8, max_length=512):
        all_embeddings = []
    
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
    
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
    
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
    
            outputs = bert_encoder(**encoded)
    
            last_hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
    
            cls_emb = last_hidden[:, 0, :]
    
            masked_hidden = last_hidden * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            mean_emb = sum_hidden / lengths
    
            emb = torch.cat([cls_emb, mean_emb], dim=1)
    
            all_embeddings.append(
                emb.detach().cpu().numpy().astype("float32")
            )
    
            if start % (batch_size * 20) == 0:
                print(f"Chunk embeddings: {start}/{len(texts)}")
    
        return np.vstack(all_embeddings)
    
    
    print("\nСчитаю chunk embeddings для train...")
    X_train_chunks = get_chunk_embeddings(
        train_chunks["chunk_text"].tolist(),
        batch_size=BERT_BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    print("\nСчитаю chunk embeddings для test...")
    X_test_chunks = get_chunk_embeddings(
        test_chunks["chunk_text"].tolist(),
        batch_size=BERT_BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    print("Train chunk embeddings:", X_train_chunks.shape)
    print("Test chunk embeddings:", X_test_chunks.shape)
    
    np.save("kaggle_train_chunk_embeddings.npy", X_train_chunks)
    np.save("kaggle_test_chunk_embeddings.npy", X_test_chunks)
    
    
    # ============================================================
    # 8. АГРЕГАЦИЯ CHUNK EMBEDDINGS ДО УРОВНЯ ВАКАНСИИ
    # ============================================================
    
    def aggregate_embeddings(chunk_embeddings, row_ids, n_rows, aggregations):
        chunk_embeddings = chunk_embeddings.astype("float32")
        row_ids = np.asarray(row_ids)
    
        dim = chunk_embeddings.shape[1]
    
        sums = np.zeros((n_rows, dim), dtype="float32")
        counts = np.zeros(n_rows, dtype="float32")
        maxs = np.full((n_rows, dim), -np.inf, dtype="float32")
    
        firsts = np.zeros((n_rows, dim), dtype="float32")
        lasts = np.zeros((n_rows, dim), dtype="float32")
        seen_first = np.zeros(n_rows, dtype=bool)
    
        for emb, row_id in zip(chunk_embeddings, row_ids):
            sums[row_id] += emb
            counts[row_id] += 1
            maxs[row_id] = np.maximum(maxs[row_id], emb)
    
            if not seen_first[row_id]:
                firsts[row_id] = emb
                seen_first[row_id] = True
    
            lasts[row_id] = emb
    
        counts_safe = np.maximum(counts, 1).reshape(-1, 1)
        means = sums / counts_safe
    
        maxs[~np.isfinite(maxs)] = 0
    
        parts = []
    
        if "mean" in aggregations:
            parts.append(means)
    
        if "max" in aggregations:
            parts.append(maxs)
    
        if "first" in aggregations:
            parts.append(firsts)
    
        if "last" in aggregations:
            parts.append(lasts)
    
        final = np.concatenate(parts, axis=1).astype("float32")
    
        return final, counts.astype("float32")
    
    
    X_train_final, train_chunk_counts = aggregate_embeddings(
        X_train_chunks,
        train_chunks["row_id"].values,
        n_rows=len(train_df),
        aggregations=AGGREGATIONS
    )
    
    X_test_final, test_chunk_counts = aggregate_embeddings(
        X_test_chunks,
        test_chunks["row_id"].values,
        n_rows=len(test_df),
        aggregations=AGGREGATIONS
    )
    
    print("Train final vacancy embeddings:", X_train_final.shape)
    print("Test final vacancy embeddings:", X_test_final.shape)
    print("=" * 80)
    
    np.save("kaggle_train_chunked_vacancy_embeddings.npy", X_train_final)
    np.save("kaggle_test_chunked_vacancy_embeddings.npy", X_test_final)
    
    
# ============================================================
# 9. ДОБАВЛЯЕМ BERT FEATURES В TRAIN / TEST
# ============================================================

chunked_bert_columns = [
    f"chunked_bert_{i}"
    for i in range(X_train_final.shape[1])
]

train_emb_df = pd.DataFrame(
    X_train_final,
    columns=chunked_bert_columns,
    index=train_df.index
)

test_emb_df = pd.DataFrame(
    X_test_final,
    columns=chunked_bert_columns,
    index=test_df.index
)

train_df = pd.concat([train_df, train_emb_df], axis=1)
test_df = pd.concat([test_df, test_emb_df], axis=1)

train_df["chunk_count"] = train_chunk_counts
test_df["chunk_count"] = test_chunk_counts

print("Train after BERT features:", train_df.shape)
print("Test after BERT features:", test_df.shape)
print("=" * 80)


# ============================================================
# 10. ДАННЫЕ ДЛЯ AUTOGLUON
# ============================================================

salary_text_feature_columns = (
    SALARY_TEXT_LEAK_FEATURE_COLUMNS
    if USE_SALARY_TEXT_LEAK_FEATURES
    else []
)

if USE_RAW_TEXT_FEATURES_IN_AUTOGLUON:
    feature_columns = (
        ["experience_from", "chunk_count"] +
        salary_text_feature_columns +
        TEXT_COLUMNS +
        ["full_text"] +
        chunked_bert_columns
    )
else:
    feature_columns = (
        ["experience_from", "chunk_count"] +
        salary_text_feature_columns +
        chunked_bert_columns
    )

train_data = train_df[[TARGET_COLUMN] + feature_columns].copy()
test_data = test_df[feature_columns].copy()

# Защита от утечек.
assert SALARY_COLUMN not in train_data.columns
assert SALARY_COLUMN not in test_data.columns
assert TARGET_COLUMN in train_data.columns
assert TARGET_COLUMN not in test_data.columns

print("AutoGluon target:", TARGET_COLUMN)
print("salary_from in train_data:", SALARY_COLUMN in train_data.columns)
print("salary_from in test_data:", SALARY_COLUMN in test_data.columns)
print("log_salary_from in train_data:", TARGET_COLUMN in train_data.columns)
print("log_salary_from in test_data:", TARGET_COLUMN in test_data.columns)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("=" * 80)


# ============================================================
# 11. AUTOGLUON НА ВСЁМ TRAIN
# ============================================================

if os.path.exists(AUTOGLOUON_PATH):
    shutil.rmtree(AUTOGLOUON_PATH)

predictor = TabularPredictor(
    label=TARGET_COLUMN,
    problem_type="regression",
    eval_metric="r2",
    path=AUTOGLOUON_PATH
).fit(
    train_data=train_data,
    presets="medium_quality",
    time_limit=AUTOGLOUON_TIME_LIMIT,
    dynamic_stacking=False,
    excluded_model_types=["RF", "XT"]
)


# ============================================================
# 12. PREDICT TEST
# ============================================================

test_pred_log_raw = predictor.predict(test_data)
test_pred_log_raw_values = test_pred_log_raw.values.astype("float32")

test_pred_log_values = test_pred_log_raw_values.copy()
postprocess_hint_count = 0

if USE_SALARY_TEXT_LEAK_FEATURES and USE_SALARY_HINT_POSTPROCESS:
    test_pred_log_values, postprocess_hint_count = apply_salary_hint_postprocess(
        test_pred_log_values,
        test_df,
        weight=SALARY_HINT_POSTPROCESS_WEIGHT,
    )

    print(
        "Salary hint postprocess applied to rows:",
        postprocess_hint_count,
        "weight:",
        SALARY_HINT_POSTPROCESS_WEIGHT,
        "min hint:",
        SALARY_HINT_POSTPROCESS_MIN,
    )

test_pred_salary = salary_from_log(test_pred_log_values)

print("Test prediction log stats:")
print(pd.Series(test_pred_log_values).describe())

if USE_SALARY_TEXT_LEAK_FEATURES and USE_SALARY_HINT_POSTPROCESS:
    print("\nRaw model prediction log stats:")
    print(pd.Series(test_pred_log_raw_values).describe())

print("\nTest prediction salary stats:")
print(pd.Series(test_pred_salary).describe())


# ============================================================
# 13. SUBMISSION
# ============================================================

submission = sample_submission.copy()

if len(submission) != len(test_pred_log_values):
    raise ValueError(
        f"Размер sample_submission ({len(submission)}) не совпадает "
        f"с размером test ({len(test_pred_log_values)})"
    )

# Пытаемся аккуратно определить колонку для предсказания.
possible_target_columns = [
    "prediction",
    "pred",
    "target",
    "label",
    "log_salary_from",
    "salary_from"
]

prediction_column = None

for col in submission.columns:
    if col.lower() in possible_target_columns:
        prediction_column = col
        break

# Если не нашли, берём последнюю колонку.
# Обычно sample_submission имеет id + target/prediction.
if prediction_column is None:
    prediction_column = submission.columns[-1]

# Целевая переменная соревнования — log_salary_from, поэтому в submission пишем log.
submission[prediction_column] = test_pred_log_values

submission.to_csv(
    SUBMISSION_PATH,
    index=False,
    encoding="utf-8-sig"
)

submission.to_csv(
    SUBMISSION_LOG_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nSubmission saved:", SUBMISSION_PATH)
print("Submission log copy saved:", SUBMISSION_LOG_PATH)
print("Prediction column:", prediction_column)
print(submission.head())


# ============================================================
# 14. DEBUG-ФАЙЛ С ПРЕДСКАЗАНИЯМИ
# ============================================================

debug_predictions = test_df[
    [
        "title",
        "location",
        "company",
        "skills",
        "experience_from",
        "chunk_count"
    ]
].copy()

if USE_SALARY_TEXT_LEAK_FEATURES:
    for col in SALARY_TEXT_LEAK_FEATURE_COLUMNS:
        debug_predictions[col] = test_df[col].values

debug_predictions["pred_log_salary_from_raw_model"] = test_pred_log_raw_values
debug_predictions["pred_log_salary_from"] = test_pred_log_values
debug_predictions["pred_salary_from"] = test_pred_salary

debug_predictions.to_csv(
    "kaggle_test_predictions_debug.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nСохранено:")
print(RUBERT_OUTPUT_DIR)
print(AUTOGLOUON_PATH)
print("kaggle_train_chunk_embeddings.npy")
print("kaggle_test_chunk_embeddings.npy")
print("kaggle_train_chunked_vacancy_embeddings.npy")
print("kaggle_test_chunked_vacancy_embeddings.npy")
print("kaggle_test_predictions_debug.csv")
print(SUBMISSION_PATH)
print(SUBMISSION_LOG_PATH)
