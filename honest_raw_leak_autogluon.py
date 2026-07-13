from autogluon.tabular import TabularPredictor

import os
import re
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
FALLBACK_SAMPLE_SUBMISSION_PATH = "sample_submition.csv"

TARGET_COLUMN = "log_salary_from"
SALARY_COLUMN = "salary_from"
TEXT_COLUMNS = ["title", "location", "company", "skills", "description"]

RANDOM_STATE = 42
VALID_SIZE = 0.2

LOCAL_MODEL_PATH = "autogluon_raw_text_salary_hints_local_valid"
FULL_MODEL_PATH = "autogluon_raw_text_salary_hints_full_train"
LOCAL_TIME_LIMIT = 700
FULL_TIME_LIMIT = 900

SUBMISSION_PATH = "submission_honest_raw_salary_hints.csv"
LOCAL_PREDICTIONS_PATH = "local_valid_raw_text_salary_hints_predictions.csv"
LOCAL_LEADERBOARD_PATH = "local_valid_raw_text_salary_hints_leaderboard.csv"


SALARY_NUMBER_PATTERN = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{5,7}|\d{2,4}(?:[,.]\d+)?)(?!\d)"
)
SALARY_KEYWORD_PATTERN = re.compile(
    r"(\u0437\u0430\u0440\u043f\u043b\u0430\u0442|"
    r"\u0437\u0430\u0440\u0430\u0431\u043e\u0442\u043d|"
    r"\u043e\u043a\u043b\u0430\u0434\u043d|"
    r"\u043e\u043a\u043b\u0430\u0434|"
    r"\u0434\u043e\u0445\u043e\u0434|"
    r"\u0437\u043f)",
    flags=re.IGNORECASE,
)
RUBLE_PATTERN = re.compile(r"(\u0440\u0443\u0431|\u20bd|\u0440\b)", flags=re.IGNORECASE)
THOUSAND_PATTERN = re.compile(r"(\u0442\u044b\u0441|k\b)", flags=re.IGNORECASE)
USD_PATTERN = re.compile(r"(\$|usd|\u0434\u043e\u043b\u043b)", flags=re.IGNORECASE)
BAD_AMOUNT_UNIT_PATTERN = re.compile(
    r"(%|/\s*\u0447|\u0432\s*\u0447\u0430\u0441|\u0447\u0430\u0441|"
    r"/\s*\u043c|\u043c2|\u043c\u00b2|"
    r"\u0437\u0430\s+\u043e\u0434\u043d|\u0437\u0430\s+\u0448\u0442|"
    r"\u0437\u0430\s+\u043f\u043b\u0430\u043d|"
    r"\u0434\u043d(?:\u0435\u0439|\u044f|\u044c)|"
    r"\u043b\u0435\u0442|\u0433\u043e\u0434|\u043c\u0435\u0441\u044f\u0446|"
    r"\u0441\u043e\u0442\u0440\u0443\u0434\u043d|"
    r"\u043c\u0435\u0442\u0440|\u043e\u0444\u0438\u0441|\u043f\u0435\u0448)",
    flags=re.IGNORECASE,
)


def add_full_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["full_text"] = (
        "Название: " + df["title"] + " [SEP] "
        "Навыки: " + df["skills"] + " [SEP] "
        "Описание: " + df["description"] + " [SEP] "
        "Регион: " + df["location"] + " [SEP] "
        "Компания: " + df["company"] + " [SEP] "
        "Опыт от: " + df["experience_from"].astype(str)
    )
    return df


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


def normalize_salary_amount(amount, context: str):
    if amount is None or amount > 1_200_000:
        return None
    if USD_PATTERN.search(context):
        amount *= 0.09
    elif amount >= 5000:
        amount /= 1000.0
    elif THOUSAND_PATTERN.search(context):
        amount = amount
    elif RUBLE_PATTERN.search(context):
        return None
    elif amount < 20:
        return None
    return float(amount) if 5 <= amount <= 1200 else None


def extract_salary_hints(text: str):
    text = re.sub(r"\s+", " ", str(text).lower().replace("\u00a0", " "))
    hints = []
    for keyword_match in SALARY_KEYWORD_PATTERN.finditer(text):
        segment = text[keyword_match.start(): min(len(text), keyword_match.start() + 170)]
        for number_match in SALARY_NUMBER_PATTERN.finditer(segment):
            local_context = segment[
                max(0, number_match.start() - 20):
                min(len(segment), number_match.end() + 35)
            ]
            raw_amount = parse_salary_number(number_match.group())
            is_large_monthly = (
                raw_amount is not None and
                raw_amount >= 5000 and
                (
                    RUBLE_PATTERN.search(local_context) or
                    THOUSAND_PATTERN.search(local_context)
                )
            )
            if BAD_AMOUNT_UNIT_PATTERN.search(local_context) and not is_large_monthly:
                continue
            amount = normalize_salary_amount(raw_amount, local_context)
            if amount is not None:
                hints.append(amount)
    unique_hints = []
    for amount in hints:
        if not any(abs(amount - known) < 0.01 for known in unique_hints):
            unique_hints.append(amount)
    return unique_hints


def add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for text in df[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1):
        hints = extract_salary_hints(text)
        if hints:
            values = np.asarray(hints, dtype="float32")
            row = {
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
            row = {
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
            value = row[col]
            row[f"{col}_log"] = float(np.log(value)) if pd.notna(value) and value > 0 else np.nan
        rows.append(row)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET_COLUMN, SALARY_COLUMN]).copy()
    train_df = train_df[train_df[SALARY_COLUMN] > 0].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    for df in [train_df, test_df]:
        for col in TEXT_COLUMNS:
            df[col] = df[col].fillna("").astype(str).str.strip()
        df["experience_from"] = df["experience_from"].fillna(0).astype("float32")
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype("float32")

    train_df = add_salary_features(add_full_text(train_df))
    test_df = add_salary_features(add_full_text(test_df))
    return train_df, test_df


def feature_columns(df: pd.DataFrame):
    ignored = {TARGET_COLUMN, SALARY_COLUMN}
    return [col for col in df.columns if col not in ignored]


def fit_predictor(path, train_data, tuning_data=None, time_limit=600):
    if os.path.exists(path):
        shutil.rmtree(path)
    return TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="regression",
        eval_metric="r2",
        path=path,
    ).fit(
        train_data=train_data,
        tuning_data=tuning_data,
        presets="medium_quality",
        time_limit=time_limit,
        dynamic_stacking=False,
        excluded_model_types=["RF", "XT", "KNN"],
    )


train_df, test_df = load_data()
cols = feature_columns(train_df)

train_df["salary_bin"] = pd.qcut(train_df[TARGET_COLUMN], q=10, duplicates="drop")
train_part, valid_part = train_test_split(
    train_df,
    test_size=VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_df["salary_bin"],
)
train_part = train_part.drop(columns=["salary_bin"]).reset_index(drop=True)
valid_part = valid_part.drop(columns=["salary_bin"]).reset_index(drop=True)
train_df = train_df.drop(columns=["salary_bin"])

print("Local train:", train_part.shape)
print("Local valid:", valid_part.shape)
print("Train salary hints:", int(train_part["salary_hint_has"].sum()))
print("Valid salary hints:", int(valid_part["salary_hint_has"].sum()))

local_predictor = fit_predictor(
    LOCAL_MODEL_PATH,
    train_part[[TARGET_COLUMN] + cols],
    tuning_data=valid_part[[TARGET_COLUMN] + cols],
    time_limit=LOCAL_TIME_LIMIT,
)
valid_pred = local_predictor.predict(valid_part[cols]).values.astype("float32")
y_valid = valid_part[TARGET_COLUMN].values.astype("float32")

print("HONEST LOCAL R2:", r2_score(y_valid, valid_pred))
print("HONEST LOCAL MAE log:", mean_absolute_error(y_valid, valid_pred))
print("HONEST LOCAL RMSE log:", mean_squared_error(y_valid, valid_pred) ** 0.5)
print(local_predictor.leaderboard(valid_part[[TARGET_COLUMN] + cols], silent=True))

valid_out = valid_part[
    ["title", "location", "company", "skills", "experience_from", SALARY_COLUMN, TARGET_COLUMN]
].copy()
valid_out["pred_log_salary_from"] = valid_pred
valid_out["pred_salary_from"] = np.exp(np.clip(valid_pred, 0, 20))
valid_out.to_csv(LOCAL_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
local_predictor.leaderboard(valid_part[[TARGET_COLUMN] + cols], silent=True).to_csv(
    LOCAL_LEADERBOARD_PATH,
    index=False,
    encoding="utf-8-sig",
)

full_predictor = fit_predictor(
    FULL_MODEL_PATH,
    train_df[[TARGET_COLUMN] + cols],
    tuning_data=None,
    time_limit=FULL_TIME_LIMIT,
)
test_pred = full_predictor.predict(test_df[cols]).values.astype("float32")

sample_path = SAMPLE_SUBMISSION_PATH if os.path.exists(SAMPLE_SUBMISSION_PATH) else FALLBACK_SAMPLE_SUBMISSION_PATH
submission = pd.read_csv(sample_path)
prediction_column = "prediction" if "prediction" in submission.columns else submission.columns[-1]
submission[prediction_column] = test_pred
submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")

print("Saved:", SUBMISSION_PATH)
print(pd.Series(test_pred).describe())
