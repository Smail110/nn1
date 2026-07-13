"""Build the final candidate-reranker submission.

The expensive base models (CatBoost/TF-IDF/transformers) are assumed to have
already produced their local OOF and test prediction files. This script
rebuilds the final stacking and salary-candidate reranking layer, then writes
the Kaggle submission.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


TARGET = "log_salary_from"
OUTPUT = Path("submission_candidate_reranker.csv")
DEBUG_OUTPUT = Path("test_candidate_reranker_debug.csv")
REPORT = Path("candidate_reranker_results.json")
USE_CONTEXT_FEATURES = False
TEXT_COLUMNS = ["title", "skills", "description", "location", "company"]
NUMBER = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{4,7}|\d{1,3}(?:[,.]\d+)?)(?!\d)"
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_safely(frame: pd.DataFrame, target: Path) -> Path:
    """Записывает CSV безопасно для Windows.

    Целевой файл может быть открыт в Excel, браузере или окне предпросмотра.
    Поэтому сначала пишем во временный файл, затем атомарно заменяем target.
    Если target заблокирован, сохраняем рядом отдельный *_FOR_UPLOAD.csv.
    """

    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    fallback = target.with_name(f"{target.stem}_FOR_UPLOAD{target.suffix}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    try:
        temporary.replace(target)
        return target
    except PermissionError:
        temporary.replace(fallback)
        print(
            f"[FILE-LOCK] {target.name} сейчас занят Windows. "
            f"Точный файл сохранён как {fallback.name}.",
            flush=True,
        )
        return fallback


def prediction(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    column = "prediction" if "prediction" in frame else frame.columns[-1]
    return frame[column].to_numpy(dtype="float64")


def aligned(path: str, valid: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != len(valid):
        raise RuntimeError(f"{path}: row count mismatch")
    if TARGET in frame and not np.allclose(frame[TARGET], valid[TARGET], atol=1e-7):
        raise RuntimeError(f"{path}: target order mismatch")
    return frame


def parse_amount(raw: str) -> float | None:
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
        value = float(cleaned)
    except ValueError:
        return None
    amount = value / 1000.0 if value >= 5000 else value
    return float(amount) if 5.0 <= amount <= 1000.0 else None


def extract_context_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_id, row in frame.reset_index(drop=True).iterrows():
        for column_id, column in enumerate(TEXT_COLUMNS):
            value = str(row.get(column, "") or "")
            parsed = []
            for match in NUMBER.finditer(value):
                amount = parse_amount(match.group())
                if amount is not None:
                    parsed.append((match, amount))
            for index, (match, amount) in enumerate(parsed):
                left = value[max(0, match.start() - 100) : match.start()].lower()
                right = value[match.end() : min(len(value), match.end() + 100)].lower()
                left_near = left[-30:]
                right_near = right[:30]
                previous_amount = parsed[index - 1][1] if index else amount
                next_amount = parsed[index + 1][1] if index + 1 < len(parsed) else amount
                previous_separator = (
                    value[parsed[index - 1][0].end() : match.start()].lower()
                    if index else ""
                )
                next_separator = (
                    value[match.end() : parsed[index + 1][0].start()].lower()
                    if index + 1 < len(parsed) else ""
                )
                money_pattern = r"₽|руб|р\.?\s*(?:/|$)|тыс|\bk\b|\$|usd|eur|евро"
                salary_pattern = r"зарплат|оклад|з/?п\b|доход|оплат|вознаграж|salary"
                bad_pattern = (
                    r"мбит|гбит|час|дн(?:ей|я)|лет|год|%|процент|сотрудник|"
                    r"кв\.?\s*м|метр|заказ|смен|проект|урок"
                )
                range_pattern = r"^\s*(?:[-–—]|до\b|по\b)"
                rows.append(
                    {
                        "row_id": row_id,
                        "amount": amount,
                        "context_column_id": column_id,
                        "context_position": match.start() / max(1, len(value)),
                        "context_text_length_log": np.log1p(len(value)),
                        "context_has_money": float(bool(re.search(money_pattern, left_near + right_near, re.I))),
                        "context_has_salary": float(bool(re.search(salary_pattern, left + right, re.I))),
                        "context_has_bad_unit": float(bool(re.search(bad_pattern, left_near + right_near, re.I))),
                        "context_preceded_from": float(bool(re.search(r"\bот\s*$", left_near, re.I))),
                        "context_preceded_to": float(bool(re.search(r"\bдо\s*$", left_near, re.I))),
                        "context_followed_period": float(bool(re.search(r"^\s*(?:руб\.?\s*)?/(?:мес|месяц|month)|в\s+месяц", right_near, re.I))),
                        "context_lower_range": float(next_amount >= amount and bool(re.search(range_pattern, next_separator, re.I))),
                        "context_upper_range": float(previous_amount <= amount and bool(re.search(range_pattern, previous_separator, re.I))),
                        "context_previous_log_gap": np.log(max(previous_amount, 1.0)) - np.log(amount),
                        "context_next_log_gap": np.log(max(next_amount, 1.0)) - np.log(amount),
                        "context_occurrence_in_column": index,
                        "context_candidates_in_column": len(parsed),
                    }
                )
    return pd.DataFrame(rows)


def build_candidate_features(
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    stack: pd.DataFrame,
    base: np.ndarray,
    experience: np.ndarray,
    include_label: bool,
) -> pd.DataFrame:
    row_id = candidates["row_id"].to_numpy(dtype="int32")
    amount = candidates["amount"].to_numpy(dtype="float64")
    amount_log = np.log(amount)
    probability = candidates["probability"].to_numpy(dtype="float64")
    features = pd.DataFrame(index=candidates.index)
    features["probability"] = probability
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    features["probability_logit"] = np.log(clipped / (1 - clipped))
    features["amount_log"] = amount_log
    features["amount"] = amount
    features["ridge_base"] = base[row_id]
    features["gap_base"] = amount_log - base[row_id]
    features["abs_gap_base"] = np.abs(features["gap_base"])
    for column in stack.columns:
        values = stack[column].to_numpy(dtype="float64")[row_id]
        features[f"gap_{column}"] = amount_log - values
        features[f"abs_gap_{column}"] = np.abs(amount_log - values)

    grouped = candidates.groupby("row_id", sort=False)
    features["row_candidate_count"] = grouped["row_id"].transform("size").to_numpy()
    features["probability_rank"] = grouped["probability"].rank(
        method="first", ascending=False
    ).to_numpy()
    features["amount_rank"] = grouped["amount"].rank(method="average").to_numpy()
    features["probability_max"] = grouped["probability"].transform("max").to_numpy()
    features["probability_mean"] = grouped["probability"].transform("mean").to_numpy()
    features["probability_share"] = probability / np.maximum(
        grouped["probability"].transform("sum").to_numpy(), 1e-8
    )
    features["distance_rank"] = features.groupby(candidates["row_id"])[
        "abs_gap_base"
    ].rank(method="first").to_numpy()
    features["duplicate_amount_count"] = candidates.groupby(
        ["row_id", "amount"]
    )["amount"].transform("size").to_numpy()
    matrix = stack.to_numpy(dtype="float64")
    features["model_std"] = matrix.std(axis=1)[row_id]
    features["model_range"] = (
        matrix.max(axis=1) - matrix.min(axis=1)
    )[row_id]
    features["experience"] = experience[row_id]
    context = extract_context_metadata(frame)
    if len(context) != len(candidates):
        raise RuntimeError(
            f"Candidate context row mismatch: {len(context)} != {len(candidates)}"
        )
    if not np.array_equal(
        context["row_id"].to_numpy(dtype="int32"), row_id
    ) or not np.allclose(
        context["amount"].to_numpy(dtype="float64"), amount
    ):
        raise RuntimeError("Candidate context order mismatch")
    context_columns = [
        column for column in context.columns
        if column not in ("row_id", "amount")
    ]
    if USE_CONTEXT_FEATURES:
        features = pd.concat(
            [features, context[context_columns].reset_index(drop=True)],
            axis=1,
        )
    if include_label and "label" not in candidates:
        raise RuntimeError("Training candidates do not contain label")
    return features


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["salary_bin"] = pd.qcut(train[TARGET], q=10, duplicates="drop")
_, valid = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["salary_bin"],
)
valid = valid.drop(columns="salary_bin").reset_index(drop=True)
train = train.drop(columns="salary_bin")
y = valid[TARGET].to_numpy(dtype="float64")

boost_local = aligned("local_valid_final_boost_081.csv", valid)
initial_local = aligned("local_valid_079_tfidf_te_predictions.csv", valid)
levels_local = aligned("local_valid_tfidf_levels_083.csv", valid)
large_local = [
    aligned(path, valid)["rubert_large_prediction"].to_numpy(dtype="float64")
    for path in (
        "local_valid_rubert_large_081.csv",
        "local_valid_rubert_large_081_seed2.csv",
        "local_valid_rubert_large_081_roberta.csv",
        "local_valid_rubert_large_081_xlm.csv",
    )
]

stack_local = pd.DataFrame(index=valid.index)
stack_local["final_boost"] = boost_local["final_boost_prediction"].to_numpy(dtype="float64")
for number, values in enumerate(large_local, start=1):
    stack_local[f"large_{number}"] = values
stack_local["autogluon_te"] = initial_local["autogluon_pred"].to_numpy(dtype="float64")
stack_local["tfidf_te"] = initial_local["extra_pred"].to_numpy(dtype="float64")
for column in levels_local.columns:
    if column.startswith("levels_"):
        stack_local[column] = levels_local[column].to_numpy(dtype="float64")

base_test = prediction("submission_081_final_boost.csv")
base_test_raw = base_test.copy()
large_test_1 = pd.read_csv("test_rubert_large_081.csv")["rubert_large_prediction"].to_numpy(dtype="float64")
large_test_2 = pd.read_csv("test_rubert_large_081_seed2.csv")["rubert_large_prediction"].to_numpy(dtype="float64")
large_test_4 = pd.read_csv("test_rubert_large_081_xlm.csv")["rubert_large_prediction"].to_numpy(dtype="float64")
roberta_surrogate = LinearRegression()
roberta_surrogate.fit(
    np.column_stack([large_local[0], large_local[1], large_local[3]]),
    large_local[2],
)
large_test_3 = roberta_surrogate.predict(
    np.column_stack([large_test_1, large_test_2, large_test_4])
)

ag_test = prediction("submission_honest_raw_salary_hints.csv")
initial_uncalibrated_test = prediction("submission_079_tfidf_te_blend_uncalibrated.csv")
tfidf_test = (initial_uncalibrated_test - 0.625 * ag_test) / 0.375
levels_test = pd.read_csv("test_tfidf_levels_087.csv")

stack_test = pd.DataFrame(index=test.index)
stack_test["final_boost"] = base_test
stack_test["large_1"] = large_test_1
stack_test["large_2"] = large_test_2
stack_test["large_3"] = large_test_3
stack_test["large_4"] = large_test_4
stack_test["autogluon_te"] = ag_test
stack_test["tfidf_te"] = tfidf_test
for column in levels_local.columns:
    if column.startswith("levels_"):
        if column not in levels_test:
            raise RuntimeError(f"Missing full test level signal: {column}")
        stack_test[column] = levels_test[column].to_numpy(dtype="float64")
if list(stack_test.columns) != list(stack_local.columns):
    raise RuntimeError("Local/test stack column mismatch")

# При полном переобучении один из ранних test-сабмитов может потерять
# калибровку, хотя его local OOF остаётся нормальным. Это особенно опасно
# для stacking: Ridge видит хороший local final_boost и даёт ему большой вес,
# а на test получает сдвинутый сигнал. Поэтому проверяем final_boost на
# распределительный дрейф относительно остальных стабильных моделей и, если
# надо, восстанавливаем test final_boost через surrogate, обученный на OOF.
stable_columns = [column for column in stack_local.columns if column != "final_boost"]
final_boost_surrogate = Ridge(alpha=1.0)
final_boost_surrogate.fit(
    stack_local[stable_columns],
    stack_local["final_boost"].to_numpy(dtype="float64"),
)
base_test_surrogate = final_boost_surrogate.predict(stack_test[stable_columns])
raw_surrogate_mean_gap = float(abs(base_test_raw.mean() - base_test_surrogate.mean()))
raw_surrogate_std_ratio = float(
    base_test_raw.std() / max(base_test_surrogate.std(), 1e-8)
)
final_boost_repaired = (
    raw_surrogate_mean_gap > 0.20
    or raw_surrogate_std_ratio < 0.70
    or raw_surrogate_std_ratio > 1.35
)
if final_boost_repaired:
    stack_test["final_boost"] = base_test_surrogate
    base_test = base_test_surrogate

ridge = Ridge(alpha=10.0)
ridge.fit(stack_local, y)
ridge_local = ridge.predict(stack_local)
ridge_test = ridge.predict(stack_test)

local_candidates = pd.read_csv("local_valid_salary_candidate_scores_083.csv")
test_candidates = pd.read_csv("test_salary_candidate_scores_087.csv")
local_features = build_candidate_features(
    local_candidates,
    valid,
    stack_local,
    ridge_local,
    valid["experience_from"].fillna(-1).to_numpy(dtype="float64"),
    include_label=True,
)
test_features = build_candidate_features(
    test_candidates,
    test,
    stack_test,
    ridge_test,
    test["experience_from"].fillna(-1).to_numpy(dtype="float64"),
    include_label=False,
)
if list(local_features.columns) != list(test_features.columns):
    raise RuntimeError("Candidate feature column mismatch")
candidate_label = local_candidates["label"].to_numpy(dtype="int8")

lgb_predictions = []
cat_predictions = []
for seed in (173, 991, 42, 2026, 31415):
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=550,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=120,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=30.0,
        verbosity=-1,
        n_jobs=8,
        random_state=seed,
    )
    lgb_model.fit(local_features, candidate_label)
    lgb_predictions.append(lgb_model.predict_proba(test_features)[:, 1])

    cat_model = CatBoostClassifier(
        iterations=550,
        depth=4,
        learning_rate=0.025,
        l2_leaf_reg=30.0,
        random_strength=1.0,
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        thread_count=8,
        random_seed=seed,
    )
    cat_model.fit(local_features, candidate_label)
    cat_predictions.append(cat_model.predict_proba(test_features)[:, 1])

meta_score = 0.5 * np.mean(lgb_predictions, axis=0) + 0.5 * np.mean(cat_predictions, axis=0)
scored = test_candidates.copy()
scored["meta_score"] = meta_score
best = scored.loc[scored.groupby("row_id")["meta_score"].idxmax()]

prediction_test = ridge_test.copy()
best_ids = best["row_id"].to_numpy(dtype="int32")
best_probability = best["probability"].to_numpy(dtype="float64")
best_log_amount = np.log(best["amount"].to_numpy(dtype="float64"))
selected = best_probability >= 0.18
selected_ids = best_ids[selected]
selected_weight = np.clip(
    1.10 * best_probability[selected],
    0.0,
    1.0,
)
prediction_test[selected_ids] += selected_weight * (
    best_log_amount[selected] - prediction_test[selected_ids]
)

sample = pd.read_csv("sample_submition.csv")
if len(sample) != len(test) or len(prediction_test) != len(test):
    raise RuntimeError("Submission row count mismatch")
prediction_column = "prediction" if "prediction" in sample else sample.columns[-1]
if not np.isfinite(prediction_test).all():
    raise RuntimeError("Non-finite final predictions")

model_prediction_test = prediction_test.copy()

sample[prediction_column] = prediction_test
written_outputs = {
    str(OUTPUT): write_csv_safely(sample, OUTPUT),
    "submission.csv": write_csv_safely(sample, Path("submission.csv")),
    "submission_final_for_upload.csv": write_csv_safely(
        sample, Path("submission_final_for_upload.csv")
    ),
}
actual_output_path = written_outputs[str(OUTPUT)]
actual_output_sha256 = sha256(actual_output_path)

debug = test[["title", "location", "company"]].copy()
debug["ridge_prediction"] = ridge_test
debug["candidate_probability"] = np.nan
debug["candidate_amount"] = np.nan
debug.loc[best_ids, "candidate_probability"] = best_probability
debug.loc[best_ids, "candidate_amount"] = best["amount"].to_numpy(dtype="float64")
debug["candidate_selected"] = False
debug.loc[selected_ids, "candidate_selected"] = True
debug["model_final_prediction"] = model_prediction_test
debug["final_prediction"] = prediction_test
debug.to_csv(DEBUG_OUTPUT, index=False, encoding="utf-8-sig")

local_best = json.loads(Path("oof_candidate_reranker_086_results.json").read_text("utf-8"))
report = {
    "local_ridge_r2": float(r2_score(y, ridge_local)),
    "local_selected_oof_r2": local_best["selected_oof_r2"],
    "local_selected_model": local_best["selected_model"],
    "ridge_coefficients": {
        column: float(value)
        for column, value in zip(stack_local.columns, ridge.coef_)
    },
    "pipeline_mode": "computed_from_intermediate_model_outputs",
    "output_sha256": actual_output_sha256,
    "written_outputs": {
        logical_name: str(actual_path)
        for logical_name, actual_path in written_outputs.items()
    },
    "roberta_prediction": "linear surrogate for the held-out transformer member",
    "candidate_threshold": 0.18,
    "candidate_weight_scale": 1.10,
    "test_candidates": len(test_candidates),
    "test_rows_with_candidates": int(test_candidates["row_id"].nunique()),
    "selected_candidate_rows": int(selected.sum()),
    "final_boost_raw_test_mean": float(base_test_raw.mean()),
    "final_boost_raw_test_std": float(base_test_raw.std()),
    "final_boost_surrogate_test_mean": float(base_test_surrogate.mean()),
    "final_boost_surrogate_test_std": float(base_test_surrogate.std()),
    "final_boost_raw_surrogate_mean_gap": raw_surrogate_mean_gap,
    "final_boost_raw_surrogate_std_ratio": raw_surrogate_std_ratio,
    "final_boost_repaired": bool(final_boost_repaired),
    "ridge_test_mean": float(ridge_test.mean()),
    "ridge_test_std": float(ridge_test.std()),
    "model_final_test_mean": float(model_prediction_test.mean()),
    "model_final_test_std": float(model_prediction_test.std()),
    "model_final_test_min": float(model_prediction_test.min()),
    "model_final_test_max": float(model_prediction_test.max()),
    "final_test_mean": float(prediction_test.mean()),
    "final_test_std": float(prediction_test.std()),
    "final_test_min": float(prediction_test.min()),
    "final_test_max": float(prediction_test.max()),
    "submission": str(actual_output_path),
    "upload_submission": str(written_outputs["submission.csv"]),
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print("Saved:", written_outputs[str(OUTPUT)], "and", written_outputs["submission.csv"], flush=True)
