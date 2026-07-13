"""Honest text/target-encoding experiments for the salary competition.

The script keeps the fixed external split used by honest_raw_leak_autogluon.py,
uses no salary_from-derived feature, and writes a new submission only when the
additional model improves the saved honest AutoGluon validation predictions.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
VALID_SIZE = 0.2
TARGET = "log_salary_from"
TEXT_COLUMNS = ["title", "location", "company", "skills", "description"]

BASE_VALID_PATH = Path("local_valid_raw_text_salary_hints_predictions.csv")
BASE_TEST_PATH = Path("submission_honest_raw_salary_hints.csv")
SAMPLE_PATHS = [Path("sample_submission.csv"), Path("sample_submition.csv")]

OUTPUT_SUBMISSION = Path("submission_079_tfidf_te_blend.csv")
OUTPUT_SUBMISSION_UNCALIBRATED = Path("submission_079_tfidf_te_blend_uncalibrated.csv")
OUTPUT_VALID = Path("local_valid_079_tfidf_te_predictions.csv")
OUTPUT_RESULTS = Path("local_079_experiment_results.json")


SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^0-9a-zа-я+#.]+", flags=re.IGNORECASE)
TITLE_NOISE_RE = re.compile(
    r"\b(?:вакансия|требуется|ищем|работа|срочно|вахта|удаленно|удалённо)\b",
    flags=re.IGNORECASE,
)
LEGAL_FORM_RE = re.compile(
    r"\b(?:ооо|оао|пао|ао|зао|ип|гк|нко|фгбу|мбу|муп|гуп)\b",
    flags=re.IGNORECASE,
)
LEVEL_RE = re.compile(
    r"\b(?:junior|middle|senior|lead|стажер|стажёр|младший|ведущий|старший|"
    r"главный|начинающий|помощник|заместитель)\b",
    flags=re.IGNORECASE,
)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().replace("ё", "е")
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_title(value: object, remove_level: bool = False) -> str:
    text = normalize_text(value)
    text = TITLE_NOISE_RE.sub(" ", text)
    if remove_level:
        text = LEVEL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_company(value: object) -> str:
    text = LEGAL_FORM_RE.sub(" ", normalize_text(value))
    return SPACE_RE.sub(" ", text).strip()


def first_location(value: object) -> str:
    text = normalize_text(value)
    return re.split(r"[,;/|]", text, maxsplit=1)[0].strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TEXT_COLUMNS:
        out[col] = out[col].fillna("").astype(str)
    out["experience_from"] = pd.to_numeric(out["experience_from"], errors="coerce").fillna(0)

    out["title_norm"] = out["title"].map(normalize_title)
    out["title_family"] = out["title"].map(lambda x: normalize_title(x, remove_level=True))
    out["company_norm"] = out["company"].map(normalize_company)
    out["location_norm"] = out["location"].map(normalize_text)
    out["location_first"] = out["location"].map(first_location)
    out["skills_norm"] = out["skills"].map(normalize_text)

    out["short_text"] = (
        "title " + out["title_norm"] + " titlefamily " + out["title_family"]
        + " company " + out["company_norm"]
        + " location " + out["location_norm"]
        + " skills " + out["skills_norm"]
    )
    out["full_text_079"] = (
        out["short_text"]
        + " description " + out["description"].map(normalize_text)
    )

    for col in ["title", "company", "skills", "description"]:
        text = out[col].fillna("").astype(str)
        out[f"{col}_chars"] = text.str.len().astype("float32")
        out[f"{col}_words"] = text.str.count(r"\S+").astype("float32")
    out["description_digits"] = out["description"].str.count(r"\d").astype("float32")
    out["description_lines"] = out["description"].str.count(r"\n").add(1).astype("float32")
    return out


GROUP_KEYS = [
    ("title_norm",),
    ("title_family",),
    ("company_norm",),
    ("location_norm",),
    ("location_first",),
    ("title_norm", "location_first"),
    ("title_family", "location_first"),
    ("company_norm", "location_first"),
    ("title_family", "company_norm"),
]


def key_series(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[list(cols)].astype(str).agg("\x1f".join, axis=1)


def target_encoding_predictions(
    train: pd.DataFrame,
    other: pd.DataFrame,
    smooth_values: tuple[float, ...] = (3.0, 10.0, 30.0),
) -> tuple[np.ndarray, list[str]]:
    global_mean = float(train[TARGET].mean())
    predictions: list[np.ndarray] = []
    names: list[str] = []
    for cols in GROUP_KEYS:
        train_key = key_series(train, cols)
        other_key = key_series(other, cols)
        stats = pd.DataFrame({"key": train_key, "target": train[TARGET].to_numpy()}).groupby("key")["target"].agg(["mean", "count"])
        for smooth in smooth_values:
            encoded = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
            predictions.append(other_key.map(encoded).fillna(global_mean).to_numpy(dtype="float32"))
            names.append("te_" + "_".join(cols) + f"_s{smooth:g}")
    return np.column_stack(predictions), names


CATEGORICAL_COLUMNS = [
    "title_norm",
    "title_family",
    "company_norm",
    "location_norm",
    "location_first",
    "skills_norm",
]
NUMERIC_COLUMNS = [
    "experience_from",
    "title_chars",
    "title_words",
    "company_chars",
    "company_words",
    "skills_chars",
    "skills_words",
    "description_chars",
    "description_words",
    "description_digits",
    "description_lines",
]


def build_sparse_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.997,
        max_features=260_000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    description_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=140_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    categories = OneHotEncoder(handle_unknown="ignore", min_frequency=2, dtype=np.float32)
    scaler = StandardScaler(with_mean=False)

    xw_train = word.fit_transform(train["full_text_079"])
    xw_valid = word.transform(valid["full_text_079"])
    xw_test = word.transform(test["full_text_079"])

    xc_train = char.fit_transform(train["short_text"])
    xc_valid = char.transform(valid["short_text"])
    xc_test = char.transform(test["short_text"])

    xd_train = description_char.fit_transform(train["description"])
    xd_valid = description_char.transform(valid["description"])
    xd_test = description_char.transform(test["description"])

    xcat_train = categories.fit_transform(train[CATEGORICAL_COLUMNS])
    xcat_valid = categories.transform(valid[CATEGORICAL_COLUMNS])
    xcat_test = categories.transform(test[CATEGORICAL_COLUMNS])

    xnum_train = scaler.fit_transform(train[NUMERIC_COLUMNS].to_numpy(dtype="float32"))
    xnum_valid = scaler.transform(valid[NUMERIC_COLUMNS].to_numpy(dtype="float32"))
    xnum_test = scaler.transform(test[NUMERIC_COLUMNS].to_numpy(dtype="float32"))

    x_train = sparse.hstack([xw_train, xc_train, xd_train, xcat_train, sparse.csr_matrix(xnum_train)], format="csr", dtype=np.float32)
    x_valid = sparse.hstack([xw_valid, xc_valid, xd_valid, xcat_valid, sparse.csr_matrix(xnum_valid)], format="csr", dtype=np.float32)
    x_test = sparse.hstack([xw_test, xc_test, xd_test, xcat_test, sparse.csr_matrix(xnum_test)], format="csr", dtype=np.float32)
    print("Sparse matrices:", x_train.shape, x_valid.shape, x_test.shape)
    return x_train, x_valid, x_test


def fit_ridges(x_train, y_train, x_valid, x_test):
    valid_predictions = []
    test_predictions = []
    scores = {}
    for alpha in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
        model.fit(x_train, y_train)
        pred_valid = model.predict(x_valid).astype("float32")
        pred_test = model.predict(x_test).astype("float32")
        valid_predictions.append(pred_valid)
        test_predictions.append(pred_test)
        scores[f"tfidf_ridge_a{alpha:g}"] = float(r2_score(y_valid_global, pred_valid))
        print(f"Ridge alpha={alpha:g}: {scores[f'tfidf_ridge_a{alpha:g}']:.6f}")
    return np.column_stack(valid_predictions), np.column_stack(test_predictions), scores


def best_two_way_blend(y, base, candidates, names):
    best = (-np.inf, None, None)
    for idx, name in enumerate(names):
        for weight in np.linspace(0.0, 1.0, 41):
            pred = (1.0 - weight) * base + weight * candidates[:, idx]
            score = r2_score(y, pred)
            if score > best[0]:
                best = (float(score), name, float(weight))
    return best


train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
train = prepare(train_raw)
test = prepare(test_raw)

train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
train_part, valid_part = train_test_split(
    train,
    test_size=VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=train["salary_bin"],
)
train_part = train_part.drop(columns=["salary_bin"]).reset_index(drop=True)
valid_part = valid_part.drop(columns=["salary_bin"]).reset_index(drop=True)
train = train.drop(columns=["salary_bin"]).reset_index(drop=True)

y_train = train_part[TARGET].to_numpy(dtype="float32")
y_valid_global = valid_part[TARGET].to_numpy(dtype="float32")

base_valid_frame = pd.read_csv(BASE_VALID_PATH)
if not np.allclose(base_valid_frame[TARGET].to_numpy(), y_valid_global, atol=1e-6):
    raise RuntimeError("Saved AutoGluon validation rows do not match the fixed split")
base_valid = base_valid_frame["pred_log_salary_from"].to_numpy(dtype="float32")

base_submission = pd.read_csv(BASE_TEST_PATH)
prediction_column = "prediction" if "prediction" in base_submission else base_submission.columns[-1]
base_test = base_submission[prediction_column].to_numpy(dtype="float32")

results = {"autogluon_base": float(r2_score(y_valid_global, base_valid))}
print("AutoGluon base:", f"{results['autogluon_base']:.6f}")

te_valid, te_names = target_encoding_predictions(train_part, valid_part)
te_test, _ = target_encoding_predictions(train, test)
for idx, name in enumerate(te_names):
    results[name] = float(r2_score(y_valid_global, te_valid[:, idx]))

x_train, x_valid, x_test = build_sparse_features(train_part, valid_part, test)
ridge_valid, ridge_test, ridge_scores = fit_ridges(x_train, y_train, x_valid, x_test)
results.update(ridge_scores)
ridge_names = list(ridge_scores)

all_valid = np.column_stack([ridge_valid, te_valid])
all_test = np.column_stack([ridge_test, te_test])
all_names = ridge_names + te_names

blend_score, blend_name, blend_weight = best_two_way_blend(
    y_valid_global, base_valid, all_valid, all_names
)
print("Best two-way blend:", blend_name, blend_weight, f"R2={blend_score:.6f}")
results["best_two_way"] = {
    "score": blend_score,
    "feature": blend_name,
    "weight": blend_weight,
}

# A compact non-negative stack. NNLS coefficients are normalized to make the
# final full-train/test blend stable and keep predictions on the target scale.
top_indices = np.argsort([results[name] for name in all_names])[-8:]
stack_valid = np.column_stack([base_valid, all_valid[:, top_indices]])
stack_test = np.column_stack([base_test, all_test[:, top_indices]])
stack_names = ["autogluon_base"] + [all_names[i] for i in top_indices]
coef, _ = nnls(np.column_stack([np.ones(len(stack_valid)), stack_valid]), y_valid_global)
intercept = float(coef[0])
weights = coef[1:]
if weights.sum() > 0:
    weights = weights / weights.sum()
    intercept = float(y_valid_global.mean() * (1.0 - weights.sum()))
stack_pred_valid = intercept + stack_valid @ weights
stack_score = float(r2_score(y_valid_global, stack_pred_valid))
print("NNLS stack:", f"R2={stack_score:.6f}", dict(zip(stack_names, weights.round(4))))
results["nnls_stack"] = {
    "score": stack_score,
    "intercept": intercept,
    "weights": {name: float(weight) for name, weight in zip(stack_names, weights)},
}

best_idx = all_names.index(blend_name)
final_valid = (1.0 - blend_weight) * base_valid + blend_weight * all_valid[:, best_idx]
if blend_name.startswith("tfidf_ridge_a"):
    selected_alpha = float(blend_name.removeprefix("tfidf_ridge_a"))
    print("Refitting selected Ridge on 100% train, alpha=", selected_alpha)
    x_full, x_test_full, _ = build_sparse_features(train, test, test.iloc[:1].copy())
    full_ridge = Ridge(alpha=selected_alpha, solver="lsqr", tol=1e-4)
    full_ridge.fit(x_full, train[TARGET].to_numpy(dtype="float32"))
    selected_test_prediction = full_ridge.predict(x_test_full).astype("float32")
else:
    selected_test_prediction = all_test[:, best_idx]

final_test_uncalibrated = (1.0 - blend_weight) * base_test + blend_weight * selected_test_prediction

# One-dimensional calibration fixes the slight variance shrinkage of the blend.
# Only two parameters are estimated on more than 3k honest validation rows.
calibration = np.polyfit(final_valid, y_valid_global, deg=1)
final_valid_calibrated = calibration[0] * final_valid + calibration[1]
final_test = calibration[0] * final_test_uncalibrated + calibration[1]
final_score = float(r2_score(y_valid_global, final_valid))
calibrated_score = float(r2_score(y_valid_global, final_valid_calibrated))

valid_output = valid_part[["title", "location", "company", "experience_from", TARGET]].copy()
valid_output["autogluon_pred"] = base_valid
valid_output["extra_pred"] = all_valid[:, best_idx]
valid_output["blend_pred"] = final_valid
valid_output["blend_pred_calibrated"] = final_valid_calibrated
valid_output.to_csv(OUTPUT_VALID, index=False, encoding="utf-8-sig")

submission = pd.read_csv(next(path for path in SAMPLE_PATHS if path.exists()))
submission[prediction_column] = final_test_uncalibrated
submission.to_csv(OUTPUT_SUBMISSION_UNCALIBRATED, index=False, encoding="utf-8-sig")
submission[prediction_column] = final_test
submission.to_csv(OUTPUT_SUBMISSION, index=False, encoding="utf-8-sig")

results["selected_submission"] = {
    "path": str(OUTPUT_SUBMISSION),
    "local_r2": final_score,
    "local_r2_calibrated": calibrated_score,
    "calibration_slope": float(calibration[0]),
    "calibration_intercept": float(calibration[1]),
    "feature": blend_name,
    "weight": blend_weight,
    "prediction_mean": float(final_test.mean()),
    "prediction_std": float(final_test.std()),
}
OUTPUT_RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("Saved:", OUTPUT_SUBMISSION)
print("Selected local R2:", f"{final_score:.6f}")
print("Calibrated local R2:", f"{calibrated_score:.6f}")
