"""Jointly tune honest BERT blending and supervised salary-hint correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


TARGET = "log_salary_from"
RAW_BERT_TEST = Path("test_full_raw_bert_081.csv")
OUTPUT = Path("submission_081_raw_bert_salary.csv")
OUTPUT_NO_HINT = Path("submission_081_raw_bert_no_hint.csv")
REPORT = Path("raw_bert_081_results.json")


def read_prediction_from_first_existing(paths: list[str]) -> np.ndarray:
    """Читает prediction из первого найденного файла.

    В ранних экспериментах один и тот же честный BERT-бленд сохранялся под
    разными именами. Для полного прогона с нуля используем актуальное имя, а
    старое оставляем только как совместимость с предыдущими запусками.
    """

    for path in paths:
        if Path(path).exists():
            return pd.read_csv(path)["prediction"].to_numpy(dtype="float64")
    raise FileNotFoundError(f"Не найден ни один файл: {paths}")

knn_local = pd.read_csv("local_valid_knn_079.csv")
bert_local = pd.read_csv("local_valid_honest_bert_079.csv")
salary_local = pd.read_csv("local_valid_supervised_salary_hint_081.csv")
y = knn_local[TARGET].to_numpy(dtype="float64")

nonbert_local = knn_local["salary_leak_knn_pred"].to_numpy(dtype="float64")
nonbert_test_frame = pd.read_csv("submission_079_salary_leak_knn.csv")
nonbert_test = nonbert_test_frame["prediction"].to_numpy(dtype="float64")
raw_bert_test = pd.read_csv(RAW_BERT_TEST)

# Recover the target predicted by the train-only supervised hint model. The
# saved candidate used 30% mapped salary and 70% old ensemble on changed rows.
old_base_local = salary_local["base_prediction"].to_numpy(dtype="float64")
salary_final_local = salary_local["final_prediction"].to_numpy(dtype="float64")
salary_mask_local = np.abs(salary_final_local - old_base_local) > 1e-10
mapped_salary_local = np.zeros(len(y), dtype="float64")
mapped_salary_local[salary_mask_local] = (
    salary_final_local[salary_mask_local] - 0.7 * old_base_local[salary_mask_local]
) / 0.3

old_base_test = read_prediction_from_first_existing(
    [
        "submission_079_honest_bert_blend.csv",
        "submission_079_final_honest_ensemble.csv",
    ]
)
salary_final_test = pd.read_csv("submission_081_supervised_salary_hint.csv")["prediction"].to_numpy(dtype="float64")
salary_mask_test = np.abs(salary_final_test - old_base_test) > 1e-10
mapped_salary_test = np.zeros(len(nonbert_test), dtype="float64")
mapped_salary_test[salary_mask_test] = (
    salary_final_test[salary_mask_test] - 0.7 * old_base_test[salary_mask_test]
) / 0.3

columns = ["bert_mean", "bert_median", "bert_first", "bert_last", "bert_min", "bert_max"]
best = (-np.inf, None, 0.0, 0.0, None)
trials = []
for column in columns:
    local_bert = bert_local[column].to_numpy(dtype="float64")
    for bert_weight in np.linspace(0.1, 0.5, 81):
        base = (1.0 - bert_weight) * nonbert_local + bert_weight * local_bert
        base_slope, base_intercept = np.polyfit(base, y, deg=1)
        base = base_slope * base + base_intercept
        for hint_weight in np.linspace(0.0, 0.5, 21):
            prediction = base.copy()
            prediction[salary_mask_local] = (
                (1.0 - hint_weight) * prediction[salary_mask_local]
                + hint_weight * mapped_salary_local[salary_mask_local]
            )
            slope, intercept = np.polyfit(prediction, y, deg=1)
            calibrated = slope * prediction + intercept
            score = float(r2_score(y, calibrated))
            trials.append((score, column, float(bert_weight), float(hint_weight)))
            if score > best[0]:
                best = (
                    score,
                    column,
                    float(bert_weight),
                    float(hint_weight),
                    (float(base_slope), float(base_intercept), float(slope), float(intercept)),
                )

score, column, bert_weight, hint_weight, calibration = best
base_slope, base_intercept, final_slope, final_intercept = calibration
print("Best:", best)

local_bert = bert_local[column].to_numpy(dtype="float64")
local_no_hint = base_slope * (
    (1.0 - bert_weight) * nonbert_local + bert_weight * local_bert
) + base_intercept
local_final = local_no_hint.copy()
local_final[salary_mask_local] = (
    (1.0 - hint_weight) * local_final[salary_mask_local]
    + hint_weight * mapped_salary_local[salary_mask_local]
)
local_final = final_slope * local_final + final_intercept

test_bert = raw_bert_test[column].to_numpy(dtype="float64")
test_no_hint = base_slope * (
    (1.0 - bert_weight) * nonbert_test + bert_weight * test_bert
) + base_intercept
test_final = test_no_hint.copy()
test_final[salary_mask_test] = (
    (1.0 - hint_weight) * test_final[salary_mask_test]
    + hint_weight * mapped_salary_test[salary_mask_test]
)
test_final = final_slope * test_final + final_intercept

no_hint_submission = nonbert_test_frame.copy()
no_hint_submission["prediction"] = test_no_hint
no_hint_submission.to_csv(OUTPUT_NO_HINT, index=False, encoding="utf-8-sig")
submission = nonbert_test_frame.copy()
submission["prediction"] = test_final
submission.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

report = {
    "local_r2": score,
    "bert_aggregation": column,
    "bert_weight": bert_weight,
    "salary_hint_weight": hint_weight,
    "base_calibration": [base_slope, base_intercept],
    "final_calibration": [final_slope, final_intercept],
    "salary_valid_rows": int(salary_mask_local.sum()),
    "salary_test_rows": int(salary_mask_test.sum()),
    "prediction_mean": float(test_final.mean()),
    "prediction_std": float(test_final.std()),
    "top_trials": [
        {
            "r2": row[0],
            "bert_aggregation": row[1],
            "bert_weight": row[2],
            "salary_hint_weight": row[3],
        }
        for row in sorted(trials, reverse=True)[:10]
    ],
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved:", OUTPUT)
