"""Learn which numeric text mentions are the actual vacancy salary_from."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "log_salary_from"
TEXT_COLUMNS = ["title", "skills", "description", "location", "company"]
NUMBER = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{4,7}|\d{1,3}(?:[,.]\d+)?)(?!\d)"
)
OUTPUT = Path("local_valid_broad_salary_candidates_083.csv")
REPORT = Path("broad_salary_candidates_083_results.json")


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
                    "has_money": float(
                        bool(re.search(r"₽|руб|р\.?\s*(?:/|$)|тыс|\bk\b|\$|usd", context, re.I))
                    ),
                    "has_salary": float(
                        bool(re.search(r"зарплат|оклад|з/?п\b|доход|оплат|вознаграж", context, re.I))
                    ),
                    "has_from": float(bool(re.search(r"\bот\s*(?:<num>|\d)", context, re.I))),
                    "has_to": float(bool(re.search(r"\bдо\s*(?:<num>|\d)", context, re.I))),
                    "bad_unit": float(
                        bool(
                            re.search(
                                r"мбит|гбит|час|дн(?:ей|я)|лет|год|%|процент|"
                                r"сотрудник|кв\.?\s*м|метр|заказ|смен",
                                context,
                                re.I,
                            )
                        )
                    ),
                }
                if include_labels:
                    target = float(row["salary_from"])
                    item["label"] = int(
                        abs(amount - target) <= max(2.0, target * 0.03)
                    )
                rows.append(item)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("No numeric candidates extracted")
    candidates["row_candidate_count"] = candidates.groupby("row_id")[
        "row_id"
    ].transform("size")
    return candidates


def numeric_matrix(frame: pd.DataFrame, scaler=None):
    columns = [
        "amount",
        "original",
        "position",
        "column_id",
        "has_money",
        "has_salary",
        "has_from",
        "has_to",
        "bad_unit",
        "row_candidate_count",
    ]
    values = frame[columns].to_numpy(dtype="float32")
    if scaler is None:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    else:
        values = scaler.transform(values)
    return csr_matrix(values.astype("float32")), scaler


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

print("Extracting candidates", flush=True)
train_candidates = extract(train_part, include_labels=True)
valid_candidates = extract(valid, include_labels=True)
print(
    "Candidates:", len(train_candidates), len(valid_candidates),
    "positive:", int(train_candidates["label"].sum()), int(valid_candidates["label"].sum()),
    flush=True,
)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=100_000,
    sublinear_tf=True,
    dtype=np.float32,
)
x_train_text = vectorizer.fit_transform(train_candidates["context"])
x_valid_text = vectorizer.transform(valid_candidates["context"])
x_train_numeric, scaler = numeric_matrix(train_candidates)
x_valid_numeric, _ = numeric_matrix(valid_candidates, scaler)
x_train = hstack([x_train_text, x_train_numeric]).tocsr()
x_valid = hstack([x_valid_text, x_valid_numeric]).tocsr()

y_candidate = train_candidates["label"].to_numpy(dtype="int8")
y_valid_candidate = valid_candidates["label"].to_numpy(dtype="int8")
y = valid[TARGET].to_numpy(dtype="float64")
base_frame = pd.read_csv("local_valid_final_boost_081.csv")
base = base_frame["final_boost_prediction"].to_numpy(dtype="float64")

best_result = None
all_results = []
for alpha in [1e-5, 3e-5, 1e-4, 3e-4]:
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=0.02,
        alpha=alpha,
        max_iter=40,
        tol=1e-4,
        class_weight=None,
        average=True,
        random_state=2026,
        n_jobs=1,
    )
    model.fit(x_train, y_candidate)
    probability = model.predict_proba(x_valid)[:, 1]
    candidate_auc = float(roc_auc_score(y_valid_candidate, probability))
    scored = valid_candidates[["row_id", "amount", "label"]].copy()
    scored["probability"] = probability
    best_per_row = scored.loc[scored.groupby("row_id")["probability"].idxmax()]
    row_probability = np.full(len(valid), np.nan)
    row_amount = np.full(len(valid), np.nan)
    row_label = np.zeros(len(valid), dtype="int8")
    ids = best_per_row["row_id"].to_numpy(dtype="int32")
    row_probability[ids] = best_per_row["probability"].to_numpy()
    row_amount[ids] = best_per_row["amount"].to_numpy()
    row_label[ids] = best_per_row["label"].to_numpy(dtype="int8")

    best = (float(r2_score(y, base)), 1.0, 0.0, 0)
    for threshold in np.linspace(0.05, 0.95, 91):
        selected = np.isfinite(row_probability) & (row_probability >= threshold)
        for weight in np.linspace(0.1, 1.0, 46):
            prediction = base.copy()
            prediction[selected] = (
                (1.0 - weight) * prediction[selected]
                + weight * np.log(row_amount[selected])
            )
            score = float(r2_score(y, prediction))
            if score > best[0]:
                best = (score, float(threshold), float(weight), int(selected.sum()))
    result = {
        "alpha": alpha,
        "candidate_auc": candidate_auc,
        "local_r2": best[0],
        "threshold": best[1],
        "weight": best[2],
        "selected_rows": best[3],
    }
    all_results.append(result)
    print(result, flush=True)
    if best_result is None or result["local_r2"] > best_result["local_r2"]:
        best_result = result
        best_result["row_probability"] = row_probability
        best_result["row_amount"] = row_amount
        best_result["row_label"] = row_label

print("Training CatBoost context classifier", flush=True)
catboost_features = [
    "context", "column", "amount_bucket", "amount", "original", "position",
    "column_id", "has_money", "has_salary", "has_from", "has_to", "bad_unit",
    "row_candidate_count",
]
cat_probabilities = []
for cat_seed in [173, 42, 2026]:
    catboost = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=12.0,
        random_strength=0.5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=cat_seed,
        verbose=200,
        allow_writing_files=False,
        thread_count=8,
    )
    catboost.fit(
        train_candidates[catboost_features],
        y_candidate,
        cat_features=["column", "amount_bucket"],
        text_features=["context"],
    )
    cat_probabilities.append(
        catboost.predict_proba(valid_candidates[catboost_features])[:, 1]
    )
cat_probability = np.mean(cat_probabilities, axis=0)
print("CatBoost candidate AUC:", roc_auc_score(y_valid_candidate, cat_probability), flush=True)
cat_scored = valid_candidates[["row_id", "amount", "label"]].copy()
cat_scored["probability"] = cat_probability
cat_scored.to_csv(
    "local_valid_salary_candidate_scores_083.csv",
    index=False,
    encoding="utf-8-sig",
)
cat_best_rows = cat_scored.loc[cat_scored.groupby("row_id")["probability"].idxmax()]
cat_row_probability = np.full(len(valid), np.nan)
cat_row_amount = np.full(len(valid), np.nan)
cat_ids = cat_best_rows["row_id"].to_numpy(dtype="int32")
cat_row_probability[cat_ids] = cat_best_rows["probability"].to_numpy()
cat_row_amount[cat_ids] = cat_best_rows["amount"].to_numpy()

output = valid[["title", "location", "company", "salary_from", TARGET]].copy()
output["candidate_probability"] = best_result.pop("row_probability")
output["candidate_amount"] = best_result.pop("row_amount")
output["candidate_exact"] = best_result.pop("row_label")
output["catboost_candidate_probability"] = cat_row_probability
output["catboost_candidate_amount"] = cat_row_amount
output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
REPORT.write_text(
    json.dumps({"best": best_result, "all": all_results}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Best:", best_result, flush=True)
