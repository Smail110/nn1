"""TF-IDF salary-level classifiers for an independent extreme-aware signal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


TARGET = "log_salary_from"
OUTPUT = Path("local_valid_tfidf_levels_083.csv")
REPORT = Path("tfidf_levels_083_results.json")


def text(frame: pd.DataFrame) -> pd.Series:
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
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, valid = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["salary_bin"],
)
train_part = train_part.drop(columns="salary_bin").reset_index(drop=True)
valid = valid.drop(columns="salary_bin").reset_index(drop=True)
train_text = text(train_part.copy())
valid_text = text(valid.copy())

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
print("Fitting TF-IDF", flush=True)
x_train = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
x_valid = hstack([word.transform(valid_text), char.transform(valid_text)]).tocsr()
print("Matrices:", x_train.shape, x_valid.shape, flush=True)

y_train = train_part[TARGET].to_numpy(dtype="float64")
y_valid = valid[TARGET].to_numpy(dtype="float64")
predictions = {}
scores = {}

for bins in [10, 15, 20]:
    edges = np.unique(np.quantile(y_train, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(np.digitize(y_train, edges[1:-1]), 0, len(edges) - 2)
    centers = np.asarray(
        [y_train[labels == index].mean() for index in range(len(edges) - 1)]
    )
    for alpha in [1e-4, 3e-4]:
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=15,
            tol=1e-4,
            random_state=2026,
            n_jobs=1,
            average=True,
        )
        model.fit(x_train, labels)
        probability = model.predict_proba(x_valid)
        prediction = probability @ centers[model.classes_]
        slope, intercept = np.polyfit(prediction, y_valid, deg=1)
        calibrated = slope * prediction + intercept
        name = f"levels_{bins}_a{alpha:g}"
        predictions[name] = calibrated
        scores[name] = float(r2_score(y_valid, calibrated))
        print(name, scores[name], flush=True)

output = valid[["title", "location", "company", TARGET]].copy()
for name, prediction in predictions.items():
    output[name] = prediction
output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
REPORT.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
print("Best:", max(scores.items(), key=lambda item: item[1]), flush=True)
