"""Finalize the hint classifier with stable OOF employer/region corrections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split


TARGET = "log_salary_from"
SEEDS = (42, 173, 991, 7, 31415)
FOLDS = 10
OUTPUT = Path("submission_081_final_boost.csv")
REPORT = Path("final_boost_081_results.json")


def normalize(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^0-9a-zа-яё+#]+", " ", regex=True)
        .str.strip()
    )


def averaged_oof_group_correction(
    key: pd.Series,
    residual: np.ndarray,
    smoothing: int,
) -> np.ndarray:
    fold_predictions = []
    for seed in SEEDS:
        correction = np.zeros(len(residual), dtype="float64")
        kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for train_index, valid_index in kfold.split(residual):
            grouped = pd.DataFrame(
                {
                    "key": key.iloc[train_index].to_numpy(),
                    "residual": residual[train_index],
                }
            ).groupby("key")["residual"].agg(["mean", "count"])
            mapping = grouped["mean"] * grouped["count"] / (
                grouped["count"] + smoothing
            )
            correction[valid_index] = (
                key.iloc[valid_index].map(mapping).fillna(0).to_numpy()
            )
        fold_predictions.append(correction)
    return np.mean(fold_predictions, axis=0)


def full_group_mapping(
    key: pd.Series,
    residual: np.ndarray,
    smoothing: int,
) -> pd.Series:
    grouped = pd.DataFrame(
        {"key": key.to_numpy(), "residual": residual}
    ).groupby("key")["residual"].agg(["mean", "count"])
    return grouped["mean"] * grouped["count"] / (
        grouped["count"] + smoothing
    )


settings = json.loads(Path("hint_classifier_081_results.json").read_text("utf-8"))
local = pd.read_csv("local_valid_hint_classifier_081.csv")
test_debug = pd.read_csv("test_hint_classifier_081.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

y = local[TARGET].to_numpy(dtype="float64")
local_prediction = local["classifier_only_prediction"].to_numpy(dtype="float64")
base_score = float(r2_score(y, local_prediction))

company_local = normalize(local["company"])
company_test = normalize(test["company"])
best_company = (base_score, 0, 0.0, np.zeros(len(local)))
for smoothing in [2, 3, 5, 10, 20]:
    correction = averaged_oof_group_correction(
        company_local, y - local_prediction, smoothing
    )
    for weight in np.linspace(0.0, 1.25, 51):
        score = float(r2_score(y, local_prediction + weight * correction))
        if score > best_company[0]:
            best_company = (score, smoothing, float(weight), correction.copy())

company_score, company_smoothing, company_weight, company_oof = best_company
local_after_company = local_prediction + company_weight * company_oof
company_mapping = full_group_mapping(
    company_local, y - local_prediction, company_smoothing
)

location_local = normalize(local["location"])
location_test = normalize(test["location"])
best_location = (
    company_score,
    0,
    0.0,
    np.zeros(len(local), dtype="float64"),
)
for smoothing in [20, 50, 100, 200, 500]:
    correction = averaged_oof_group_correction(
        location_local, y - local_after_company, smoothing
    )
    for weight in np.linspace(0.0, 1.25, 51):
        score = float(r2_score(y, local_after_company + weight * correction))
        if score > best_location[0]:
            best_location = (score, smoothing, float(weight), correction.copy())

location_score, location_smoothing, location_weight, location_oof = best_location
local_prediction = local_after_company + location_weight * location_oof
location_mapping = full_group_mapping(
    location_local, y - local_after_company, location_smoothing
)

slope, intercept = np.polyfit(local_prediction, y, deg=1)
local_prediction = slope * local_prediction + intercept
calibrated_score = float(r2_score(y, local_prediction))

train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, _ = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["salary_bin"],
)
best_snap = (calibrated_score, 0, 0.0, local_prediction)
for minimum_count in [2, 3, 5, 10, 20]:
    counts = train_part["salary_from"].value_counts()
    levels = np.log(np.sort(counts[counts >= minimum_count].index.to_numpy()))
    nearest = levels[
        np.abs(local_prediction[:, None] - levels[None, :]).argmin(axis=1)
    ]
    for weight in np.linspace(0.0, 1.0, 101):
        snapped = (1.0 - weight) * local_prediction + weight * nearest
        score = float(r2_score(y, snapped))
        if score > best_snap[0]:
            best_snap = (score, minimum_count, float(weight), snapped)

final_score, snap_minimum_count, snap_weight, final_local = best_snap

submission = pd.read_csv("submission_081_two_bert.csv")
prediction_column = "prediction" if "prediction" in submission else submission.columns[-1]
test_prediction = submission[prediction_column].to_numpy(dtype="float64")
probability = test_debug["hint_exact_probability"].to_numpy(dtype="float64")
hint = test_debug["hint"].to_numpy(dtype="float64")
confidence = test_debug["hint_confidence"].to_numpy(dtype="int32")
selected = np.isfinite(probability) & (probability >= settings["threshold"])
weights = np.where(
    confidence[selected] >= 5,
    settings["high_confidence_hint_weight"],
    settings["other_hint_weight"],
)
test_prediction[selected] = (
    (1.0 - weights) * test_prediction[selected]
    + weights * np.log(hint[selected])
)

test_prediction += company_weight * (
    company_test.map(company_mapping).fillna(0).to_numpy()
)
test_prediction += location_weight * (
    location_test.map(location_mapping).fillna(0).to_numpy()
)
test_prediction = slope * test_prediction + intercept

if snap_weight > 0:
    scaled_count = max(
        1,
        int(round(snap_minimum_count * len(train) / len(train_part))),
    )
    counts = train["salary_from"].value_counts()
    levels = np.log(np.sort(counts[counts >= scaled_count].index.to_numpy()))
    nearest = levels[
        np.abs(test_prediction[:, None] - levels[None, :]).argmin(axis=1)
    ]
    test_prediction = (1.0 - snap_weight) * test_prediction + snap_weight * nearest

submission[prediction_column] = test_prediction
submission.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")

local_debug = local.copy()
local_debug["final_boost_prediction"] = final_local
local_debug.to_csv("local_valid_final_boost_081.csv", index=False, encoding="utf-8-sig")

report = {
    "classifier_local_r2": base_score,
    "company_local_r2": company_score,
    "location_local_r2": location_score,
    "calibrated_local_r2": calibrated_score,
    "final_local_r2": final_score,
    "company_smoothing": company_smoothing,
    "company_weight": company_weight,
    "location_smoothing": location_smoothing,
    "location_weight": location_weight,
    "calibration": [float(slope), float(intercept)],
    "snap_minimum_count": snap_minimum_count,
    "snap_weight": snap_weight,
    "selected_test_hints": int(selected.sum()),
    "prediction_mean": float(test_prediction.mean()),
    "prediction_std": float(test_prediction.std()),
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved:", OUTPUT, "and submission.csv")
