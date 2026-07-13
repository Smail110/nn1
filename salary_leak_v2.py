"""Evaluate and apply high-confidence salary amounts explicitly present in text."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


TARGET = "log_salary_from"
RANDOM_STATE = 42
TEXT_COLUMNS = ["title", "location", "company", "skills", "description"]
VALID_PREDICTIONS = Path("local_valid_079_tfidf_te_predictions.csv")
TEST_PREDICTIONS = Path("submission_079_tfidf_te_blend.csv")
OUTPUT = Path("submission_079_salary_leak_v2.csv")
LOCAL_OUTPUT = Path("local_valid_salary_leak_v2.csv")
TEST_HINTS_OUTPUT = Path("test_salary_hints_v2.csv")
REPORT = Path("salary_leak_v2_results.json")

NUMBER_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{4,7}|\d{2,3}(?:[,.]\d+)?)(?!\d)"
)
SALARY_RE = re.compile(r"зарплат|з/?п\b|оклад|доход|оплат|ставк|вознагражд", re.I)
MONEY_RE = re.compile(r"₽|руб|\bтыс|т\.\s*р\b|\bk\b|\$|usd|доллар|€|eur|евро", re.I)
THOUSAND_RE = re.compile(r"\bтыс|т\.\s*р\b|\bk\b", re.I)
USD_RE = re.compile(r"\$|usd|доллар", re.I)
EUR_RE = re.compile(r"€|eur|евро", re.I)
BAD_PERIOD_RE = re.compile(
    r"%|процент|в\s+час|/\s*час|часов|за\s+час|в\s+день|за\s+смен|"
    r"в\s+недел|за\s+заказ|за\s+выход|кв\.\s*м|м[²2]",
    re.I,
)
FROM_BEFORE_RE = re.compile(r"(?:\bот|свыше|начиная\s+с)\s*$", re.I)
TO_BEFORE_RE = re.compile(r"(?:\bдо|не\s+более)\s*$", re.I)
RANGE_AFTER_RE = re.compile(r"^\s*(?:[-–—]|до)\s*\d", re.I)


def parse_number(raw: str) -> float | None:
    text = raw.replace("\u00a0", " ").strip()
    grouped = bool(
        re.search(r"\d[\s.]\d{3}\b", text)
        or re.search(r"\d,\d{3}\b", text)
    )
    text = re.sub(r"(?<=\d)\s(?=\d)", "", text)
    if grouped:
        text = text.replace(".", "").replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_amount(raw_amount: float | None, context: str) -> float | None:
    if raw_amount is None or raw_amount <= 0 or raw_amount > 2_000_000:
        return None
    if 1900 <= raw_amount <= 2035 and not MONEY_RE.search(context):
        return None
    if USD_RE.search(context):
        amount = raw_amount * 0.09
    elif EUR_RE.search(context):
        amount = raw_amount * 0.10
    elif raw_amount >= 5_000:
        amount = raw_amount / 1_000.0
    elif THOUSAND_RE.search(context):
        amount = raw_amount
    elif re.search(r"₽|руб", context, re.I) and SALARY_RE.search(context):
        amount = raw_amount
    else:
        return None
    return float(amount) if 5.0 <= amount <= 1_000.0 else None


def extract_best_hint(text: str) -> tuple[float, int, int]:
    text = re.sub(r"\s+", " ", str(text).lower().replace("\u00a0", " "))
    candidates = []
    for match in NUMBER_RE.finditer(text):
        left = text[max(0, match.start() - 70) : match.start()]
        right = text[match.end() : min(len(text), match.end() + 70)]
        context = left + match.group() + right
        if BAD_PERIOD_RE.search(context):
            continue
        if not (MONEY_RE.search(context) or SALARY_RE.search(context)):
            continue
        amount = normalize_amount(parse_number(match.group()), context)
        if amount is None:
            continue

        priority = 1
        if FROM_BEFORE_RE.search(left[-25:]):
            priority = 5
        elif RANGE_AFTER_RE.search(right[:25]):
            priority = 4
        elif SALARY_RE.search(context) and MONEY_RE.search(context):
            priority = 3
        elif TO_BEFORE_RE.search(left[-25:]):
            priority = 1
        elif MONEY_RE.search(context):
            priority = 2
        candidates.append((amount, priority, match.start()))

    if not candidates:
        return np.nan, 0, 0
    max_priority = max(item[1] for item in candidates)
    strongest = [item for item in candidates if item[1] == max_priority]
    # For ranges and multiple salary statements, salary_from is the lower bound.
    best = min(strongest, key=lambda item: (item[0], item[2]))
    return best[0], best[1], len(candidates)


def add_hints(df: pd.DataFrame) -> pd.DataFrame:
    text = df[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)
    parsed = [extract_best_hint(value) for value in text]
    return pd.DataFrame(parsed, columns=["hint", "confidence", "hint_count"])


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
_, valid = train_test_split(
    train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train["salary_bin"],
)
valid = valid.drop(columns="salary_bin").reset_index(drop=True)

valid_hints = add_hints(valid)
test_hints = add_hints(test)
test_hints.to_csv(TEST_HINTS_OUTPUT, index=False, encoding="utf-8-sig")
saved_valid = pd.read_csv(VALID_PREDICTIONS)
if not np.allclose(saved_valid[TARGET], valid[TARGET], atol=1e-6):
    raise RuntimeError("Validation order mismatch")

y = valid[TARGET].to_numpy(dtype="float32")
base = saved_valid["blend_pred_calibrated"].to_numpy(dtype="float32")
hint_log = np.log(valid_hints["hint"].to_numpy(dtype="float32"))
test_hint_log = np.log(test_hints["hint"].to_numpy(dtype="float32"))

best = (float(r2_score(y, base)), 0, 0.0, 0.0)
rows = []
for min_confidence in [1, 2, 3, 4, 5]:
    for min_amount in [20.0, 35.0, 50.0, 70.0]:
        mask = (
            (valid_hints["confidence"].to_numpy() >= min_confidence)
            & (valid_hints["hint"].to_numpy() >= min_amount)
        )
        for weight in np.linspace(0.025, 0.5, 20):
            pred = base.copy()
            pred[mask] = (1.0 - weight) * pred[mask] + weight * hint_log[mask]
            score = float(r2_score(y, pred))
            rows.append((score, min_confidence, min_amount, float(weight), int(mask.sum())))
            if score > best[0]:
                best = (score, min_confidence, min_amount, float(weight))

score, min_confidence, min_amount, weight = best

# Let explicit lower bounds (confidence 5) and ordinary salary mentions
# (confidence 3-4) have separate shrinkage strengths.
valid_high = (valid_hints["confidence"].to_numpy() >= 5) & (valid_hints["hint"].to_numpy() >= 20)
valid_medium = (
    (valid_hints["confidence"].to_numpy() >= 3)
    & (valid_hints["confidence"].to_numpy() < 5)
    & (valid_hints["hint"].to_numpy() >= 20)
)
tier_best = (float(r2_score(y, base)), 0.0, 0.0)
for high_weight in np.linspace(0.0, 0.6, 25):
    for medium_weight in np.linspace(0.0, 0.6, 25):
        pred = base.copy()
        pred[valid_high] = (1.0 - high_weight) * pred[valid_high] + high_weight * hint_log[valid_high]
        pred[valid_medium] = (1.0 - medium_weight) * pred[valid_medium] + medium_weight * hint_log[valid_medium]
        tier_score = float(r2_score(y, pred))
        if tier_score > tier_best[0]:
            tier_best = (tier_score, float(high_weight), float(medium_weight))

if tier_best[0] >= score:
    score, high_weight, medium_weight = tier_best
    valid_mask = valid_high | valid_medium
    test_high = (test_hints["confidence"].to_numpy() >= 5) & (test_hints["hint"].to_numpy() >= 20)
    test_medium = (
        (test_hints["confidence"].to_numpy() >= 3)
        & (test_hints["confidence"].to_numpy() < 5)
        & (test_hints["hint"].to_numpy() >= 20)
    )
    test_mask = test_high | test_medium
else:
    high_weight = weight
    medium_weight = weight
    valid_mask = (
        (valid_hints["confidence"].to_numpy() >= min_confidence)
        & (valid_hints["hint"].to_numpy() >= min_amount)
    )
    test_mask = (
        (test_hints["confidence"].to_numpy() >= min_confidence)
        & (test_hints["hint"].to_numpy() >= min_amount)
    )
    test_high = test_mask
    test_medium = np.zeros(len(test), dtype=bool)

test_submission = pd.read_csv(TEST_PREDICTIONS)
prediction_column = "prediction" if "prediction" in test_submission else test_submission.columns[-1]
test_pred = test_submission[prediction_column].to_numpy(dtype="float32")
test_pred[test_high] = (
    (1.0 - high_weight) * test_pred[test_high] + high_weight * test_hint_log[test_high]
)
test_pred[test_medium] = (
    (1.0 - medium_weight) * test_pred[test_medium] + medium_weight * test_hint_log[test_medium]
)
test_submission[prediction_column] = test_pred
test_submission.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

valid_pred = base.copy()
if tier_best[0] >= best[0]:
    valid_pred[valid_high] = (
        (1.0 - high_weight) * valid_pred[valid_high] + high_weight * hint_log[valid_high]
    )
    valid_pred[valid_medium] = (
        (1.0 - medium_weight) * valid_pred[valid_medium] + medium_weight * hint_log[valid_medium]
    )
else:
    valid_pred[valid_mask] = (
        (1.0 - weight) * valid_pred[valid_mask] + weight * hint_log[valid_mask]
    )
local_output = saved_valid.copy()
local_output["salary_hint"] = valid_hints["hint"]
local_output["salary_hint_confidence"] = valid_hints["confidence"]
local_output["salary_leak_pred"] = valid_pred
local_output.to_csv(LOCAL_OUTPUT, index=False, encoding="utf-8-sig")

true_salary = valid["salary_from"].to_numpy(dtype="float32")
has_hint = valid_hints["hint"].notna().to_numpy()
exact = has_hint & (np.abs(valid_hints["hint"].to_numpy() - true_salary) <= np.maximum(2.0, true_salary * 0.03))
report = {
    "base_r2": float(r2_score(y, base)),
    "best_r2": score,
    "min_confidence": int(min_confidence),
    "min_amount": float(min_amount),
    "weight": float(weight),
    "high_confidence_weight": float(high_weight),
    "medium_confidence_weight": float(medium_weight),
    "valid_rows_changed": int(valid_mask.sum()),
    "test_rows_changed": int(test_mask.sum()),
    "valid_hint_coverage": float(has_hint.mean()),
    "valid_exact_target_coverage": float(exact.mean()),
    "top_trials": [
        {
            "r2": item[0],
            "min_confidence": item[1],
            "min_amount": item[2],
            "weight": item[3],
            "rows": item[4],
        }
        for item in sorted(rows, reverse=True)[:10]
    ],
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved:", OUTPUT)
