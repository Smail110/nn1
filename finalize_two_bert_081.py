"""Blend the raw chunked BERT ensemble with the second full 512-token BERT."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


TARGET = "log_salary_from"

# Rebuild the current best honest local prediction.
knn = pd.read_csv("local_valid_knn_079.csv")
bert = pd.read_csv("local_valid_honest_bert_079.csv")
salary = pd.read_csv("local_valid_supervised_salary_hint_081.csv")
raw_report = json.load(open("raw_bert_081_results.json", encoding="utf-8"))
y = knn[TARGET].to_numpy(dtype="float64")

current = raw_report["base_calibration"][0] * (
    (1.0 - raw_report["bert_weight"]) * knn["salary_leak_knn_pred"].to_numpy(dtype="float64")
    + raw_report["bert_weight"] * bert[raw_report["bert_aggregation"]].to_numpy(dtype="float64")
) + raw_report["base_calibration"][1]

salary_mask = np.abs(salary["final_prediction"] - salary["base_prediction"]) > 1e-10
mapped_salary = np.zeros(len(y), dtype="float64")
mapped_salary[salary_mask] = (
    salary.loc[salary_mask, "final_prediction"].to_numpy()
    - 0.7 * salary.loc[salary_mask, "base_prediction"].to_numpy()
) / 0.3
current[salary_mask] = (
    (1.0 - raw_report["salary_hint_weight"]) * current[salary_mask]
    + raw_report["salary_hint_weight"] * mapped_salary[salary_mask]
)
current = raw_report["final_calibration"][0] * current + raw_report["final_calibration"][1]

# The saved local checkpoint predicts standardized target values.
train = pd.read_csv("train.csv")
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, _ = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["salary_bin"],
)
second_local_standardized = pd.read_csv("local_valid_second_bert_081.csv")[
    "second_bert_prediction"
].to_numpy(dtype="float64")
second_local = (
    second_local_standardized * train_part[TARGET].std()
    + train_part[TARGET].mean()
)

best = (-np.inf, 0.0, 1.0, 0.0)
for weight in np.linspace(0.0, 0.4, 161):
    prediction = (1.0 - weight) * current + weight * second_local
    slope, intercept = np.polyfit(prediction, y, deg=1)
    calibrated = slope * prediction + intercept
    score = float(r2_score(y, calibrated))
    if score > best[0]:
        best = (score, float(weight), float(slope), float(intercept))

score, weight, slope, intercept = best
base_test_frame = pd.read_csv("submission_081_raw_bert_salary.csv")
base_test = base_test_frame["prediction"].to_numpy(dtype="float64")
second_test = pd.read_csv("test_full_second_bert_081.csv")[
    "second_bert_prediction"
].to_numpy(dtype="float64")
final_test = slope * ((1.0 - weight) * base_test + weight * second_test) + intercept

submission = base_test_frame.copy()
submission["prediction"] = final_test
submission.to_csv("submission_081_two_bert.csv", index=False, encoding="utf-8-sig")

report = {
    "local_r2": score,
    "second_bert_weight": weight,
    "calibration": [slope, intercept],
    "prediction_mean": float(final_test.mean()),
    "prediction_std": float(final_test.std()),
}
open("two_bert_081_results.json", "w", encoding="utf-8").write(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved: submission_081_two_bert.csv")
