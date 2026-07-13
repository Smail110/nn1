"""Leakage-free nearest-vacancy salary signal and blend evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


TARGET = "log_salary_from"
RANDOM_STATE = 42
LOCAL_BASE = Path("local_valid_salary_leak_v2.csv")
TEST_BASE = Path("submission_079_salary_leak_v2.csv")
LOCAL_OUTPUT = Path("local_valid_knn_079.csv")
OUTPUT = Path("submission_079_salary_leak_knn.csv")
REPORT = Path("knn_079_results.json")


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я+#.]+", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def prepare_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").map(normalize)
    company = df["company"].fillna("").map(normalize)
    location = df["location"].fillna("").map(normalize)
    skills = df["skills"].fillna("").map(normalize)
    return (
        "title " + title + " title " + title
        + " company " + company
        + " location " + location
        + " skills " + skills
    )


def build_features(train_text, valid_text, test_text):
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=160_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    xw_train = word.fit_transform(train_text)
    xw_valid = word.transform(valid_text)
    xw_test = word.transform(test_text)
    xc_train = char.fit_transform(train_text)
    xc_valid = char.transform(valid_text)
    xc_test = char.transform(test_text)
    return (
        sparse.hstack([xw_train, xc_train], format="csr"),
        sparse.hstack([xw_valid, xc_valid], format="csr"),
        sparse.hstack([xw_test, xc_test], format="csr"),
    )


def query_neighbors(x_train, x_other, neighbors=50):
    model = NearestNeighbors(
        n_neighbors=neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    ).fit(x_train)
    distances, indices = model.kneighbors(x_other, return_distance=True)
    return np.clip(1.0 - distances, 0.0, 1.0), indices


def neighbor_prediction(similarity, indices, targets, k, power):
    sim = similarity[:, :k] ** power
    values = targets[indices[:, :k]]
    denominator = sim.sum(axis=1)
    global_mean = float(targets.mean())
    return np.divide(
        (sim * values).sum(axis=1),
        denominator,
        out=np.full(len(sim), global_mean, dtype="float32"),
        where=denominator > 1e-8,
    ).astype("float32")


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

x_train, x_valid, x_test = build_features(
    prepare_text(train_part), prepare_text(valid_part), prepare_text(test)
)
print("Matrices:", x_train.shape, x_valid.shape, x_test.shape)
valid_similarity, valid_indices = query_neighbors(x_train, x_valid)
y_train = train_part[TARGET].to_numpy(dtype="float32")
y_valid = valid_part[TARGET].to_numpy(dtype="float32")

local_base = pd.read_csv(LOCAL_BASE)
if not np.allclose(local_base[TARGET], y_valid, atol=1e-6):
    raise RuntimeError("Validation order mismatch")
base_valid = local_base["salary_leak_pred"].to_numpy(dtype="float32")

best = (float(r2_score(y_valid, base_valid)), None, None, 0.0)
scores = {}
valid_candidates = {}
for k in [3, 5, 10, 20, 35, 50]:
    for power in [1.0, 2.0, 4.0]:
        name = f"knn_k{k}_p{power:g}"
        pred = neighbor_prediction(valid_similarity, valid_indices, y_train, k, power)
        valid_candidates[name] = pred
        scores[name] = float(r2_score(y_valid, pred))
        for weight in np.linspace(0.0, 0.3, 31):
            blend = (1.0 - weight) * base_valid + weight * pred
            score = float(r2_score(y_valid, blend))
            if score > best[0]:
                best = (score, k, power, float(weight))
print("Best:", best)

score, best_k, best_power, best_weight = best
if best_k is None:
    raise RuntimeError("KNN did not improve the base model")
best_name = f"knn_k{best_k}_p{best_power:g}"
best_valid_knn = valid_candidates[best_name]

# Refit the retrieval vocabulary and index on every labeled row.
x_full, x_test_full, _ = build_features(
    prepare_text(train), prepare_text(test), prepare_text(test.iloc[:1])
)
test_similarity, test_indices = query_neighbors(x_full, x_test_full)
test_knn = neighbor_prediction(
    test_similarity,
    test_indices,
    train[TARGET].to_numpy(dtype="float32"),
    best_k,
    best_power,
)

test_submission = pd.read_csv(TEST_BASE)
prediction_column = "prediction" if "prediction" in test_submission else test_submission.columns[-1]
base_test = test_submission[prediction_column].to_numpy(dtype="float32")
final_test = (1.0 - best_weight) * base_test + best_weight * test_knn
test_submission[prediction_column] = final_test
test_submission.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

local_output = local_base.copy()
local_output["knn_pred"] = best_valid_knn
local_output["salary_leak_knn_pred"] = (
    (1.0 - best_weight) * base_valid + best_weight * best_valid_knn
)
local_output.to_csv(LOCAL_OUTPUT, index=False, encoding="utf-8-sig")

report = {
    "base_r2": float(r2_score(y_valid, base_valid)),
    "best_r2": score,
    "k": int(best_k),
    "power": float(best_power),
    "weight": best_weight,
    "standalone_scores": scores,
    "mean_top_similarity_valid": float(valid_similarity[:, 0].mean()),
    "mean_top_similarity_test": float(test_similarity[:, 0].mean()),
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved:", OUTPUT)
