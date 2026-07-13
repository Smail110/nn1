"""Fit the selected salary-level TF-IDF signals on full train and score test."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


TARGET = "log_salary_from"
OUTPUT = Path("test_tfidf_levels_087.csv")
REPORT = Path("full_tfidf_levels_087_results.json")


def text(frame: pd.DataFrame) -> pd.Series:
    frame = frame.copy()
    for column in ["title", "skills", "location", "company", "description"]:
        frame[column] = frame[column].fillna("").astype(str).str.lower().str.strip()
    return (
        "title " + frame["title"]
        + " skills " + frame["skills"]
        + " location " + frame["location"]
        + " company " + frame["company"]
        + " description " + frame["description"]
    )


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
local = pd.read_csv("local_valid_tfidf_levels_083.csv")
settings = []
for column in local.columns:
    match = re.fullmatch(r"levels_(\d+)_a(.+)", column)
    if match:
        settings.append((column, int(match.group(1)), float(match.group(2))))
if not settings:
    raise RuntimeError("No selected local TF-IDF level columns found")

word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=80_000,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32,
)
char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=80_000,
    sublinear_tf=True,
    dtype=np.float32,
)
train_text = text(train)
test_text = text(test)
print("Fitting full TF-IDF", flush=True)
x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
x_test = hstack([word.transform(test_text), char.transform(test_text)]).tocsr()
print("Matrices:", x_train.shape, x_test.shape, flush=True)

y = train[TARGET].to_numpy(dtype="float64")
output = pd.DataFrame(index=np.arange(len(test)))
report = {}
for name, bins, alpha in settings:
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(np.digitize(y, edges[1:-1]), 0, len(edges) - 2)
    centers = np.asarray([y[labels == index].mean() for index in range(len(edges) - 1)])
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=25,
        tol=1e-4,
        random_state=2026,
        n_jobs=1,
        average=True,
    )
    model.fit(x_train, labels)
    raw = model.predict_proba(x_test) @ centers[model.classes_]

    # Local signals were linearly calibrated on the holdout.  Distribution
    # matching transfers that scale to the full-train model without test labels.
    local_values = local[name].to_numpy(dtype="float64")
    raw_std = float(raw.std())
    if raw_std <= 1e-8:
        calibrated = np.full(len(raw), float(local_values.mean()))
    else:
        calibrated = (
            (raw - raw.mean()) / raw_std * local_values.std()
            + local_values.mean()
        )
    output[name] = calibrated
    report[name] = {
        "bins": bins,
        "alpha": alpha,
        "raw_mean": float(raw.mean()),
        "raw_std": raw_std,
        "output_mean": float(calibrated.mean()),
        "output_std": float(calibrated.std()),
    }
    print(name, report[name], flush=True)

output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", OUTPUT, REPORT, flush=True)
