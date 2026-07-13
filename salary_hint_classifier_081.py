"""Boost the two-BERT ensemble with a supervised salary-hint reliability model.

The text parser finds explicit money amounts.  A classifier then decides when
the extracted amount is likely to be the vacancy's actual ``salary_from``.
This avoids the main failure mode of a blind regex post-process: prices,
bonuses, annual revenue and upper range bounds that merely look like salaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

import salary_leak_v2 as salary_parser


TARGET = "log_salary_from"
RANDOM_STATE = 42
BASE_SUBMISSION = Path("submission_081_two_bert.csv")
OUTPUT = Path("submission_081_hint_classifier.csv")
LOCAL_OUTPUT = Path("local_valid_hint_classifier_081.csv")
REPORT = Path("hint_classifier_081_results.json")
SEEDS = (42, 173)


def normalize_text(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^0-9a-zа-яё+#]+", " ", regex=True)
        .str.strip()
    )


def build_features(
    frame: pd.DataFrame,
    hints: pd.DataFrame,
    hint_frequencies: pd.Series,
) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(frame)))
    hint = hints["hint"].fillna(1.0).to_numpy(dtype="float64")
    result["log_hint"] = np.log(np.maximum(hint, 1.0))
    result["hint"] = hint
    result["confidence"] = hints["confidence"].to_numpy(dtype="float64")
    result["hint_count"] = hints["hint_count"].to_numpy(dtype="float64")
    result["hint_frequency"] = (
        hints["hint"].map(hint_frequencies).fillna(0).to_numpy(dtype="float64")
    )
    result["hint_bucket"] = (
        np.round(hint / 5.0) * 5.0
    ).astype("int32").astype(str)
    result["experience"] = (
        pd.to_numeric(frame["experience_from"], errors="coerce")
        .fillna(0)
        .astype(str)
        .to_numpy()
    )
    for column in ["title", "company", "location"]:
        result[column] = normalize_text(frame[column]).to_numpy()

    description = frame["description"].fillna("").astype(str).str.lower()
    result["has_range"] = description.str.contains(
        r"\bот\s+\d|\d\s*[-–—]\s*\d|\bдо\s+\d", regex=True
    ).astype(str)
    result["has_salary_word"] = description.str.contains(
        r"зарплат|оклад|оплат|доход|вознаграж", regex=True
    ).astype(str)
    result["has_net_word"] = description.str.contains(
        r"на руки|после вычета|net\b|gross\b|до вычета", regex=True
    ).astype(str)
    result["has_bonus_word"] = description.str.contains(
        r"бонус|преми|процент|kpi|мотивац", regex=True
    ).astype(str)
    result["description_chars"] = description.str.len().to_numpy(dtype="float64")
    return result


CATEGORICAL_FEATURES = [
    "hint_bucket",
    "experience",
    "title",
    "company",
    "location",
    "has_range",
    "has_salary_word",
    "has_net_word",
    "has_bonus_word",
]


def fit_classifier_ensemble(
    features: pd.DataFrame, labels: np.ndarray
) -> list[CatBoostClassifier]:
    models = []
    for seed in SEEDS:
        model = CatBoostClassifier(
            iterations=650,
            depth=5,
            learning_rate=0.035,
            l2_leaf_reg=15.0,
            random_strength=0.7,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        )
        model.fit(features, labels, cat_features=CATEGORICAL_FEATURES)
        models.append(model)
    return models


def predict_probability(
    models: list[CatBoostClassifier], features: pd.DataFrame
) -> np.ndarray:
    return np.mean(
        [model.predict_proba(features)[:, 1] for model in models], axis=0
    )


def reconstruct_local_base(train_part: pd.DataFrame) -> np.ndarray:
    knn = pd.read_csv("local_valid_knn_079.csv")
    bert = pd.read_csv("local_valid_honest_bert_079.csv")
    salary = pd.read_csv("local_valid_supervised_salary_hint_081.csv")
    raw_report = json.loads(Path("raw_bert_081_results.json").read_text("utf-8"))

    prediction = raw_report["base_calibration"][0] * (
        (1.0 - raw_report["bert_weight"])
        * knn["salary_leak_knn_pred"].to_numpy(dtype="float64")
        + raw_report["bert_weight"]
        * bert[raw_report["bert_aggregation"]].to_numpy(dtype="float64")
    ) + raw_report["base_calibration"][1]

    changed = np.abs(
        salary["final_prediction"].to_numpy()
        - salary["base_prediction"].to_numpy()
    ) > 1e-10
    mapped_salary = np.zeros(len(prediction), dtype="float64")
    mapped_salary[changed] = (
        salary.loc[changed, "final_prediction"].to_numpy(dtype="float64")
        - 0.7 * salary.loc[changed, "base_prediction"].to_numpy(dtype="float64")
    ) / 0.3
    prediction[changed] = (
        (1.0 - raw_report["salary_hint_weight"]) * prediction[changed]
        + raw_report["salary_hint_weight"] * mapped_salary[changed]
    )
    prediction = (
        raw_report["final_calibration"][0] * prediction
        + raw_report["final_calibration"][1]
    )

    second_report = json.loads(Path("two_bert_081_results.json").read_text("utf-8"))
    second_standardized = pd.read_csv("local_valid_second_bert_081.csv")[
        "second_bert_prediction"
    ].to_numpy(dtype="float64")
    second = (
        second_standardized * train_part[TARGET].std()
        + train_part[TARGET].mean()
    )
    prediction = second_report["calibration"][0] * (
        (1.0 - second_report["second_bert_weight"]) * prediction
        + second_report["second_bert_weight"] * second
    ) + second_report["calibration"][1]
    return prediction


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, valid = train_test_split(
    train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train["salary_bin"],
)
train_part = train_part.drop(columns="salary_bin").reset_index(drop=True)
valid = valid.drop(columns="salary_bin").reset_index(drop=True)
full_train = train.drop(columns="salary_bin").reset_index(drop=True)

print("Extracting salary hints", flush=True)
train_hints = salary_parser.add_hints(train_part)
valid_hints = salary_parser.add_hints(valid)
full_hints = salary_parser.add_hints(full_train)
test_hints = salary_parser.add_hints(test)

train_mask = train_hints["hint"].notna().to_numpy()
valid_mask = valid_hints["hint"].notna().to_numpy()
full_mask = full_hints["hint"].notna().to_numpy()
test_mask = test_hints["hint"].notna().to_numpy()

train_frequencies = train_hints.loc[train_mask, "hint"].value_counts()
full_frequencies = full_hints.loc[full_mask, "hint"].value_counts()
train_features = build_features(train_part, train_hints, train_frequencies).loc[
    train_mask
]
valid_features = build_features(valid, valid_hints, train_frequencies).loc[
    valid_mask
]
full_features = build_features(full_train, full_hints, full_frequencies).loc[
    full_mask
]
test_features = build_features(test, test_hints, full_frequencies).loc[test_mask]

train_exact = (
    np.abs(
        train_hints.loc[train_mask, "hint"].to_numpy(dtype="float64")
        - train_part.loc[train_mask, "salary_from"].to_numpy(dtype="float64")
    )
    <= np.maximum(
        2.0,
        0.03 * train_part.loc[train_mask, "salary_from"].to_numpy(dtype="float64"),
    )
).astype("int8")
valid_exact = (
    np.abs(
        valid_hints.loc[valid_mask, "hint"].to_numpy(dtype="float64")
        - valid.loc[valid_mask, "salary_from"].to_numpy(dtype="float64")
    )
    <= np.maximum(
        2.0,
        0.03 * valid.loc[valid_mask, "salary_from"].to_numpy(dtype="float64"),
    )
).astype("int8")
full_exact = (
    np.abs(
        full_hints.loc[full_mask, "hint"].to_numpy(dtype="float64")
        - full_train.loc[full_mask, "salary_from"].to_numpy(dtype="float64")
    )
    <= np.maximum(
        2.0,
        0.03 * full_train.loc[full_mask, "salary_from"].to_numpy(dtype="float64"),
    )
).astype("int8")

local_models = fit_classifier_ensemble(train_features, train_exact)
valid_probability = predict_probability(local_models, valid_features)
auc = float(roc_auc_score(valid_exact, valid_probability))
print("Hint reliability AUC:", auc, flush=True)

y = valid[TARGET].to_numpy(dtype="float64")
base_local = reconstruct_local_base(train_part)
base_score = float(r2_score(y, base_local))
valid_indices = np.flatnonzero(valid_mask)
valid_hint_log = np.log(
    valid_hints.loc[valid_mask, "hint"].to_numpy(dtype="float64")
)

best = (base_score, 1.0, 0.0, 0.0, 0)
trials = []
valid_hint_confidence = valid_hints.loc[
    valid_mask, "confidence"
].to_numpy(dtype="int32")
for threshold in np.linspace(0.35, 0.80, 19):
    selected = valid_probability >= threshold
    selected_indices = valid_indices[selected]
    for high_weight in np.linspace(0.55, 1.0, 19):
        for other_weight in np.linspace(0.20, 0.80, 25):
            row_weights = np.where(
                valid_hint_confidence[selected] >= 5,
                high_weight,
                other_weight,
            )
            prediction = base_local.copy()
            prediction[selected_indices] = (
                (1.0 - row_weights) * prediction[selected_indices]
                + row_weights * valid_hint_log[selected]
            )
            score = float(r2_score(y, prediction))
            trials.append(
                (
                    score,
                    float(threshold),
                    float(high_weight),
                    float(other_weight),
                    int(selected.sum()),
                )
            )
            if score > best[0]:
                best = (
                    score,
                    float(threshold),
                    float(high_weight),
                    float(other_weight),
                    int(selected.sum()),
                )

score, threshold, high_weight, other_weight, selected_valid_rows = best
selected = valid_probability >= threshold
final_local = base_local.copy()
selected_indices = valid_indices[selected]
selected_local_weights = np.where(
    valid_hint_confidence[selected] >= 5, high_weight, other_weight
)
final_local[selected_indices] = (
    (1.0 - selected_local_weights) * final_local[selected_indices]
    + selected_local_weights * valid_hint_log[selected]
)

# Company-specific residual means correct a small but repeatable bias.  The
# smoothing strength is selected on out-of-fold corrections, not on fitted
# residuals, so rare employers cannot memorize the validation labels.
company_valid = normalize_text(valid["company"])
kfold = KFold(n_splits=10, shuffle=True, random_state=173)
best_company = (float(r2_score(y, final_local)), 0, 0.0, np.zeros(len(y)))
for smoothing in [3, 5, 10, 20, 50]:
    oof_correction = np.zeros(len(y), dtype="float64")
    for train_index, holdout_index in kfold.split(valid):
        grouped = pd.DataFrame(
            {
                "company": company_valid.iloc[train_index].to_numpy(),
                "residual": y[train_index] - final_local[train_index],
            }
        ).groupby("company")["residual"].agg(["mean", "count"])
        mapping = grouped["mean"] * grouped["count"] / (
            grouped["count"] + smoothing
        )
        oof_correction[holdout_index] = (
            company_valid.iloc[holdout_index].map(mapping).fillna(0).to_numpy()
        )
    for correction_weight in np.linspace(0.0, 1.25, 26):
        corrected = final_local + correction_weight * oof_correction
        company_score = float(r2_score(y, corrected))
        if company_score > best_company[0]:
            best_company = (
                company_score,
                smoothing,
                float(correction_weight),
                oof_correction.copy(),
            )

company_score, company_smoothing, company_weight, oof_company = best_company
final_local = final_local + company_weight * oof_company
measured_company_score = float(r2_score(y, final_local))
if not np.isclose(measured_company_score, company_score, atol=1e-12):
    raise RuntimeError(
        f"Company correction mismatch: stored={company_score}, "
        f"measured={measured_company_score}"
    )

# Keep the same global scale calibration used throughout the previous ensemble.
slope, intercept = np.polyfit(final_local, y, deg=1)
calibrated_local = slope * final_local + intercept
calibrated_score = float(r2_score(y, calibrated_local))

# The target contains rounded salary levels.  Snapping is accepted only when it
# improves the holdout, and the allowed levels themselves come from train_part.
best_snap = (calibrated_score, 0, 0.0, calibrated_local)
for minimum_count in [2, 3, 5, 10, 20]:
    counts = train_part["salary_from"].value_counts()
    levels = np.log(np.sort(counts[counts >= minimum_count].index.to_numpy()))
    nearest = levels[
        np.abs(calibrated_local[:, None] - levels[None, :]).argmin(axis=1)
    ]
    for snap_weight in np.linspace(0.0, 1.0, 101):
        snapped = (
            (1.0 - snap_weight) * calibrated_local + snap_weight * nearest
        )
        snap_score = float(r2_score(y, snapped))
        if snap_score > best_snap[0]:
            best_snap = (
                snap_score,
                minimum_count,
                float(snap_weight),
                snapped,
            )

final_score, snap_minimum_count, snap_weight, calibrated_local = best_snap

print(
    "Local R2:", base_score, "->", score, "->", company_score,
    "->", calibrated_score, "->", final_score,
    "threshold/weights:", threshold, high_weight, other_weight,
    flush=True,
)

full_models = fit_classifier_ensemble(full_features, full_exact)
test_probability = predict_probability(full_models, test_features)
selected_test = test_probability >= threshold
test_indices = np.flatnonzero(test_mask)
selected_test_indices = test_indices[selected_test]
test_hint_log = np.log(
    test_hints.loc[test_mask, "hint"].to_numpy(dtype="float64")
)
test_hint_confidence = test_hints.loc[
    test_mask, "confidence"
].to_numpy(dtype="int32")

test_probability_all = np.full(len(test), np.nan)
test_probability_all[test_mask] = test_probability
test_debug = test[["title", "location", "company"]].copy()
test_debug["hint"] = test_hints["hint"]
test_debug["hint_confidence"] = test_hints["confidence"]
test_debug["hint_exact_probability"] = test_probability_all
test_debug.to_csv("test_hint_classifier_081.csv", index=False, encoding="utf-8-sig")

submission = pd.read_csv(BASE_SUBMISSION)
prediction_column = "prediction" if "prediction" in submission else submission.columns[-1]
test_prediction = submission[prediction_column].to_numpy(dtype="float64")
selected_test_weights = np.where(
    test_hint_confidence[selected_test] >= 5, high_weight, other_weight
)
test_prediction[selected_test_indices] = (
    (1.0 - selected_test_weights) * test_prediction[selected_test_indices]
    + selected_test_weights * test_hint_log[selected_test]
)

full_company_residual = pd.DataFrame(
    {
        "company": company_valid.to_numpy(),
        "residual": y - (
            # Recreate the classifier-only prediction before OOF company correction.
            base_local
        ),
    }
)
# Replace residuals on selected rows with residuals after the hint correction.
classifier_only_local = base_local.copy()
classifier_only_local[selected_indices] = (
    (1.0 - selected_local_weights) * classifier_only_local[selected_indices]
    + selected_local_weights * valid_hint_log[selected]
)
full_company_residual["residual"] = y - classifier_only_local
grouped = full_company_residual.groupby("company")["residual"].agg(
    ["mean", "count"]
)
company_mapping = grouped["mean"] * grouped["count"] / (
    grouped["count"] + company_smoothing
)
test_prediction += company_weight * (
    normalize_text(test["company"]).map(company_mapping).fillna(0).to_numpy()
)
test_prediction = slope * test_prediction + intercept

if snap_weight > 0:
    scaled_minimum_count = max(
        1,
        int(round(snap_minimum_count * len(full_train) / len(train_part))),
    )
    counts = full_train["salary_from"].value_counts()
    levels = np.log(
        np.sort(counts[counts >= scaled_minimum_count].index.to_numpy())
    )
    nearest = levels[
        np.abs(test_prediction[:, None] - levels[None, :]).argmin(axis=1)
    ]
    test_prediction = (
        (1.0 - snap_weight) * test_prediction + snap_weight * nearest
    )
submission[prediction_column] = test_prediction
submission.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")

local_output = valid[["title", "location", "company", "salary_from", TARGET]].copy()
local_output["base_prediction"] = base_local
local_output["hint"] = valid_hints["hint"]
local_output["hint_confidence"] = valid_hints["confidence"]
local_probability = np.full(len(valid), np.nan)
local_probability[valid_mask] = valid_probability
local_output["hint_exact_probability"] = local_probability
classifier_only_debug = base_local.copy()
classifier_only_debug[selected_indices] = (
    (1.0 - selected_local_weights) * classifier_only_debug[selected_indices]
    + selected_local_weights * valid_hint_log[selected]
)
local_output["classifier_only_prediction"] = classifier_only_debug
local_output["company_oof_correction"] = oof_company
local_output["final_prediction"] = calibrated_local
local_output.to_csv(LOCAL_OUTPUT, index=False, encoding="utf-8-sig")

report = {
    "base_local_r2": base_score,
    "hint_classifier_auc": auc,
    "hint_corrected_local_r2": score,
    "company_corrected_local_r2": company_score,
    "calibrated_local_r2": calibrated_score,
    "final_local_r2": final_score,
    "threshold": threshold,
    "high_confidence_hint_weight": high_weight,
    "other_hint_weight": other_weight,
    "calibration": [float(slope), float(intercept)],
    "company_smoothing": int(company_smoothing),
    "company_correction_weight": company_weight,
    "snap_minimum_count": int(snap_minimum_count),
    "snap_weight": snap_weight,
    "valid_hint_rows": int(valid_mask.sum()),
    "selected_valid_rows": selected_valid_rows,
    "selected_valid_precision": float(valid_exact[selected].mean()),
    "test_hint_rows": int(test_mask.sum()),
    "selected_test_rows": int(selected_test.sum()),
    "prediction_mean": float(test_prediction.mean()),
    "prediction_std": float(test_prediction.std()),
    "top_trials": [
        {
            "r2": item[0],
            "threshold": item[1],
            "high_weight": item[2],
            "other_weight": item[3],
            "rows": item[4],
        }
        for item in sorted(trials, reverse=True)[:10]
    ],
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print("Saved:", OUTPUT, "and submission.csv", flush=True)
