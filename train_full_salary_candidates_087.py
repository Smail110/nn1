"""Train the numeric-context classifier on full train and score test candidates."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


TARGET = "log_salary_from"
TEXT_COLUMNS = ["title", "skills", "description", "location", "company"]
NUMBER = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{4,7}|\d{1,3}(?:[,.]\d+)?)(?!\d)"
)
OUTPUT = Path("test_salary_candidate_scores_087.csv")
REPORT = Path("full_salary_candidates_087_results.json")


def parse_amount(raw: str) -> tuple[float | None, float | None]:
    raw = raw.replace("\u00a0", " ").strip()
    grouped = bool(
        re.search(r"\d[\s.]\d{3}\b", raw)
        or re.search(r"\d,\d{3}\b", raw)
    )
    cleaned = re.sub(r"(?<=\d)\s(?=\d)", "", raw)
    if grouped:
        cleaned = cleaned.replace(".", "").replace(",", "")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        original = float(cleaned)
    except ValueError:
        return None, None
    amount = original / 1000.0 if original >= 5000 else original
    if not 5.0 <= amount <= 1000.0:
        return None, original
    return float(amount), original


def extract(frame: pd.DataFrame, include_labels: bool) -> pd.DataFrame:
    rows = []
    for row_id, row in frame.reset_index(drop=True).iterrows():
        for column in TEXT_COLUMNS:
            value = str(row.get(column, "") or "")
            for match in NUMBER.finditer(value):
                amount, original = parse_amount(match.group())
                if amount is None:
                    continue
                left = value[max(0, match.start() - 90) : match.start()]
                right = value[match.end() : min(len(value), match.end() + 90)]
                context = (left + " <NUM> " + right).lower()
                amount_bin = int(round(amount / 5.0) * 5)
                context = (
                    f"column_{column} amount_{amount_bin} "
                    f"position_{min(9, match.start() // 250)} " + context
                )
                item = {
                    "row_id": row_id,
                    "amount": amount,
                    "original": original,
                    "context": context,
                    "column": column,
                    "amount_bucket": str(amount_bin),
                    "position": match.start() / max(1, len(value)),
                    "column_id": TEXT_COLUMNS.index(column),
                    "has_money": float(bool(re.search(r"₽|руб|тыс|\bk\b|\$|usd", context, re.I))),
                    "has_salary": float(bool(re.search(r"зарплат|оклад|з/?п\b|доход|оплат|вознаграж", context, re.I))),
                    "has_from": float(bool(re.search(r"\bот\s*(?:<num>|\d)", context, re.I))),
                    "has_to": float(bool(re.search(r"\bдо\s*(?:<num>|\d)", context, re.I))),
                    "bad_unit": float(bool(re.search(r"мбит|гбит|час|дн(?:ей|я)|лет|год|%|процент|сотрудник|кв\.?\s*м|метр|заказ|смен", context, re.I))),
                }
                if include_labels:
                    target = float(row["salary_from"])
                    item["label"] = int(abs(amount - target) <= max(2.0, target * 0.03))
                rows.append(item)
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No numeric candidates extracted")
    result["row_candidate_count"] = result.groupby("row_id")["row_id"].transform("size")
    return result


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Extracting full train/test candidates", flush=True)
train_candidates = extract(train, include_labels=True)
test_candidates = extract(test, include_labels=False)
print(
    "Candidates:", len(train_candidates), len(test_candidates),
    "train positives:", int(train_candidates["label"].sum()),
    flush=True,
)

feature_columns = [
    "context", "column", "amount_bucket", "amount", "original", "position",
    "column_id", "has_money", "has_salary", "has_from", "has_to", "bad_unit",
    "row_candidate_count",
]
probabilities = []
for seed in (173, 42, 2026):
    model = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=12.0,
        random_strength=0.5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=200,
        allow_writing_files=False,
        thread_count=8,
    )
    model.fit(
        train_candidates[feature_columns],
        train_candidates["label"].to_numpy(dtype="int8"),
        cat_features=["column", "amount_bucket"],
        text_features=["context"],
    )
    probabilities.append(model.predict_proba(test_candidates[feature_columns])[:, 1])

test_candidates["probability"] = np.mean(probabilities, axis=0)
test_candidates[["row_id", "amount", "probability"]].to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)
report = {
    "train_candidates": len(train_candidates),
    "train_positive_candidates": int(train_candidates["label"].sum()),
    "test_candidates": len(test_candidates),
    "test_rows_with_candidates": int(test_candidates["row_id"].nunique()),
    "probability_mean": float(test_candidates["probability"].mean()),
    "probability_max": float(test_candidates["probability"].max()),
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print("Saved:", OUTPUT, REPORT, flush=True)
