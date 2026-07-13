"""Supervised calibration of explicit salary mentions, evaluated honestly."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import salary_leak_v2 as salary_parser


TARGET = "log_salary_from"
RANDOM_STATE = 42
OUTPUT = Path("submission_081_supervised_salary_hint.csv")
LOCAL_OUTPUT = Path("local_valid_supervised_salary_hint_081.csv")
REPORT = Path("supervised_salary_hint_081_results.json")


def current_local_prediction():
    knn = pd.read_csv("local_valid_knn_079.csv")
    bert = pd.read_csv("local_valid_honest_bert_079.csv")
    before = (
        0.69 * knn["salary_leak_knn_pred"].to_numpy(dtype="float64")
        + 0.31 * bert["bert_first"].to_numpy(dtype="float64")
    )
    return 1.0017938728513054 * before - 0.02474683669425746


def build_features(frame: pd.DataFrame, hints: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(frame)))
    output["log_hint"] = np.log(np.maximum(hints["hint"].to_numpy(dtype="float64"), 1))
    output["confidence"] = hints["confidence"].to_numpy(dtype="float64")
    output["hint_count"] = hints["hint_count"].to_numpy(dtype="float64")
    output["experience"] = frame["experience_from"].to_numpy(dtype="float64")
    output["title_chars"] = frame["title"].fillna("").astype(str).str.len().to_numpy()
    output["description_chars"] = frame["description"].fillna("").astype(str).str.len().to_numpy()
    output["description_words"] = frame["description"].fillna("").astype(str).str.count(r"\S+").to_numpy()
    output["skills_missing"] = frame["skills"].fillna("").astype(str).str.strip().eq("").astype("float64").to_numpy()
    for level in range(1, 6):
        output[f"confidence_{level}"] = (hints["confidence"].to_numpy() == level).astype("float64")
    return output


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, valid_part = train_test_split(
    train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train["salary_bin"],
)
train_part = train_part.drop(columns="salary_bin").reset_index(drop=True)
valid_part = valid_part.drop(columns="salary_bin").reset_index(drop=True)
train = train.drop(columns="salary_bin").reset_index(drop=True)

print("Extracting train hints")
train_hints = salary_parser.add_hints(train_part)
valid_hints = salary_parser.add_hints(valid_part)
test_hints = salary_parser.add_hints(test)
full_hints = salary_parser.add_hints(train)

train_mask = train_hints["hint"].notna().to_numpy() & (train_hints["hint"].to_numpy() >= 20)
valid_mask = valid_hints["hint"].notna().to_numpy() & (valid_hints["hint"].to_numpy() >= 20)
test_mask = test_hints["hint"].notna().to_numpy() & (test_hints["hint"].to_numpy() >= 20)
full_mask = full_hints["hint"].notna().to_numpy() & (full_hints["hint"].to_numpy() >= 20)

x_train_all = build_features(train_part, train_hints)
x_valid_all = build_features(valid_part, valid_hints)
x_test_all = build_features(test, test_hints)
x_full_all = build_features(train, full_hints)
x_train = x_train_all.loc[train_mask]
x_valid = x_valid_all.loc[valid_mask]
y_train = train_part.loc[train_mask, TARGET].to_numpy(dtype="float64")
y_valid = valid_part[TARGET].to_numpy(dtype="float64")

base_valid = current_local_prediction()
base_test_path = (
    "submission_079_final_honest_ensemble.csv"
    if os.path.exists("submission_079_final_honest_ensemble.csv")
    else "submission_079_honest_bert_blend.csv"
)
base_test = pd.read_csv(base_test_path)["prediction"].to_numpy(dtype="float64")
models = {
    "extra_trees": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=1,
        random_state=RANDOM_STATE,
    ),
    "catboost": CatBoostRegressor(
        iterations=600,
        depth=5,
        learning_rate=0.035,
        loss_function="RMSE",
        l2_leaf_reg=12,
        random_strength=0.5,
        verbose=False,
        allow_writing_files=False,
        random_seed=RANDOM_STATE,
    ),
}

results = {}
valid_model_predictions = {}
best = (float(r2_score(y_valid, base_valid)), None, 0.0, 0)
for name, model in models.items():
    model.fit(x_train, y_train)
    hint_prediction = model.predict(x_valid)
    full_prediction = base_valid.copy()
    full_prediction[valid_mask] = hint_prediction
    valid_model_predictions[name] = full_prediction
    results[name + "_replace"] = float(r2_score(y_valid, full_prediction))

    for min_confidence in [1, 2, 3, 4, 5]:
        mask = valid_mask & (valid_hints["confidence"].to_numpy() >= min_confidence)
        mapped = np.zeros(len(valid_part), dtype="float64")
        mapped[valid_mask] = hint_prediction
        for weight in np.linspace(0.0, 0.8, 33):
            prediction = base_valid.copy()
            prediction[mask] = (
                (1.0 - weight) * prediction[mask] + weight * mapped[mask]
            )
            score = float(r2_score(y_valid, prediction))
            if score > best[0]:
                best = (score, name, float(weight), min_confidence)
    print(name, results[name + "_replace"])

best_score, best_name, best_weight, min_confidence = best
print("Best:", best)
if best_name is None:
    raise RuntimeError("Supervised hint calibration did not improve current ensemble")

final_model = models[best_name]
final_model.fit(x_full_all.loc[full_mask], train.loc[full_mask, TARGET].to_numpy(dtype="float64"))
test_hint_prediction = final_model.predict(x_test_all.loc[test_mask])
test_mapped = np.zeros(len(test), dtype="float64")
test_mapped[test_mask] = test_hint_prediction
test_apply_mask = test_mask & (test_hints["confidence"].to_numpy() >= min_confidence)
final_test = base_test.copy()
final_test[test_apply_mask] = (
    (1.0 - best_weight) * final_test[test_apply_mask]
    + best_weight * test_mapped[test_apply_mask]
)

sample = pd.read_csv("sample_submition.csv")
sample["prediction"] = final_test
sample.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

best_valid_hint = valid_model_predictions[best_name]
final_valid = base_valid.copy()
valid_apply_mask = valid_mask & (valid_hints["confidence"].to_numpy() >= min_confidence)
final_valid[valid_apply_mask] = (
    (1.0 - best_weight) * final_valid[valid_apply_mask]
    + best_weight * best_valid_hint[valid_apply_mask]
)
local_output = valid_part[["title", "location", "company", TARGET]].copy()
local_output["base_prediction"] = base_valid
local_output["hint"] = valid_hints["hint"]
local_output["confidence"] = valid_hints["confidence"]
local_output["final_prediction"] = final_valid
local_output.to_csv(LOCAL_OUTPUT, index=False, encoding="utf-8-sig")

report = {
    "base_r2": float(r2_score(y_valid, base_valid)),
    "replace_scores": results,
    "best_r2": best_score,
    "selected_model": best_name,
    "weight": best_weight,
    "min_confidence": int(min_confidence),
    "valid_rows": int(valid_apply_mask.sum()),
    "test_rows": int(test_apply_mask.sum()),
    "prediction_mean": float(final_test.mean()),
    "prediction_std": float(final_test.std()),
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved:", OUTPUT)
