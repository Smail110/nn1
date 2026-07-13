"""Реестр стадий полного leaderboard pipeline.

Каждая стадия — отдельный файл в `stage_scripts/`. Главный orchestrator
`run_all.py` запускает их по порядку, проверяет входы/выходы и пишет логи.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from config import PIPELINE_DIR, ROOT


@dataclass
class Stage:
    """Описание одной стадии pipeline."""

    name: str
    script: str
    outputs: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    note: str = ""
    heavy: bool = False

    @property
    def command(self) -> list[str]:
        return [sys.executable, "-u", str(PIPELINE_DIR / "stage_scripts" / self.script)]

    def output_paths(self) -> list[Path]:
        return [ROOT / item for item in self.outputs]

    def input_paths(self) -> list[Path]:
        return [ROOT / item for item in self.required_inputs]

    def is_done(self) -> bool:
        paths = self.output_paths()
        return bool(paths) and all(path.exists() for path in paths)


STAGES: list[Stage] = [
    Stage(
        "00_check_inputs",
        "00_check_inputs.py",
        outputs=["artifacts/00_inputs_ok.json"],
        required_inputs=["train.csv", "test.csv", "sample_submition.csv"],
        note="Проверка исходных CSV и локальных open pretrained моделей.",
    ),
    Stage(
        "01_autogluon_raw_text_salary_hints",
        "01_autogluon_raw_text_salary_hints.py",
        outputs=["local_valid_raw_text_salary_hints_predictions.csv", "submission_honest_raw_salary_hints.csv"],
        required_inputs=["train.csv", "test.csv"],
        note="AutoGluon/text baseline с salary-hint признаками.",
        heavy=True,
    ),
    Stage(
        "02_tfidf_target_encoding_079",
        "02_tfidf_target_encoding_079.py",
        outputs=[
            "local_valid_079_tfidf_te_predictions.csv",
            "submission_079_tfidf_te_blend.csv",
            "submission_079_tfidf_te_blend_uncalibrated.csv",
            "local_079_experiment_results.json",
        ],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_raw_text_salary_hints_predictions.csv",
            "submission_honest_raw_salary_hints.csv",
        ],
        note="TF-IDF + target encoding stack-сигналы.",
    ),
    Stage(
        "03_salary_leak_v2",
        "03_salary_leak_v2.py",
        outputs=["local_valid_salary_leak_v2.csv", "submission_079_salary_leak_v2.csv", "test_salary_hints_v2.csv"],
        required_inputs=["train.csv", "test.csv"],
        note="Regex salary hints baseline.",
    ),
    Stage(
        "04_knn_text_blend_079",
        "04_knn_text_blend_079.py",
        outputs=["local_valid_knn_079.csv", "submission_079_salary_leak_knn.csv"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_salary_leak_v2.csv",
            "submission_079_salary_leak_v2.csv",
        ],
        note="KNN/text correction.",
    ),
    Stage(
        "05_train_chunked_full_rubert_079",
        "05_train_chunked_full_rubert_079.py",
        outputs=["rubert_salary_chunked_full_train"],
        required_inputs=["train.csv", "test.csv"],
        note="Тяжёлая full-train chunked RuBERT стадия.",
        heavy=True,
    ),
    Stage(
        "06_honest_chunked_bert_blend_079",
        "06_honest_chunked_bert_blend_079.py",
        outputs=[
            "local_valid_honest_bert_079.csv",
            "test_honest_full_bert_079.csv",
            "submission_079_honest_bert_blend.csv",
            "submission_079_honest_bert_blend_uncalibrated.csv",
        ],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_079_tfidf_te_predictions.csv",
            "submission_079_tfidf_te_blend.csv",
            "rubert_salary_chunked_full_train",
        ],
        note="Chunked RuBERT validation/test predictions.",
        heavy=True,
    ),
    Stage(
        "07_supervised_salary_hint_081",
        "07_supervised_salary_hint_081.py",
        outputs=["local_valid_supervised_salary_hint_081.csv", "submission_081_supervised_salary_hint.csv", "supervised_salary_hint_081_results.json"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_honest_bert_079.csv",
            "submission_079_honest_bert_blend.csv",
        ],
        note="Supervised salary hint calibration.",
    ),
    Stage(
        "08_train_full_raw_bert_081",
        "08_train_full_raw_bert_081.py",
        outputs=["test_full_raw_bert_081.csv", "rubert_salary_full_raw_081"],
        required_inputs=["train.csv", "test.csv"],
        note="Full-data raw-target RuBERT.",
        heavy=True,
    ),
    Stage(
        "09_finalize_raw_bert_081",
        "09_finalize_raw_bert_081.py",
        outputs=["submission_081_raw_bert_salary.csv", "submission_081_raw_bert_no_hint.csv", "raw_bert_081_results.json"],
        required_inputs=[
            "local_valid_knn_079.csv",
            "local_valid_honest_bert_079.csv",
            "local_valid_supervised_salary_hint_081.csv",
            "test_full_raw_bert_081.csv",
        ],
        note="Сборка raw BERT 081.",
    ),
    Stage(
        "10_train_second_bert_local_valid_081",
        "10_train_second_bert_local_valid_081.py",
        outputs=["rubert_salary_local_valid_only"],
        required_inputs=["train.csv"],
        note="С нуля обучает local-valid BERT модель для честного OOF.",
        heavy=True,
    ),
    Stage(
        "11_evaluate_second_bert_local_081",
        "10_evaluate_second_bert_local_081.py",
        outputs=["local_valid_second_bert_081.csv"],
        required_inputs=["train.csv", "rubert_salary_local_valid_only"],
        note="Оценка сохранённой local BERT модели.",
        heavy=True,
    ),
    Stage(
        "12_train_full_second_bert_081",
        "11_train_full_second_bert_081.py",
        outputs=["test_full_second_bert_081.csv", "rubert_salary_full_second_081"],
        required_inputs=["train.csv", "test.csv"],
        note="Второй full BERT.",
        heavy=True,
    ),
    Stage(
        "13_finalize_two_bert_081",
        "12_finalize_two_bert_081.py",
        outputs=["submission_081_two_bert.csv", "two_bert_081_results.json"],
        required_inputs=[
            "local_valid_knn_079.csv",
            "local_valid_honest_bert_079.csv",
            "local_valid_supervised_salary_hint_081.csv",
            "raw_bert_081_results.json",
            "local_valid_second_bert_081.csv",
            "test_full_second_bert_081.csv",
            "submission_081_raw_bert_salary.csv",
        ],
        note="Blend двух BERT моделей.",
    ),
    Stage(
        "14_salary_hint_classifier_081",
        "13_salary_hint_classifier_081.py",
        outputs=["local_valid_hint_classifier_081.csv", "test_hint_classifier_081.csv", "submission_081_hint_classifier.csv", "hint_classifier_081_results.json"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "submission_081_two_bert.csv",
            "local_valid_knn_079.csv",
            "local_valid_honest_bert_079.csv",
            "local_valid_supervised_salary_hint_081.csv",
            "raw_bert_081_results.json",
            "two_bert_081_results.json",
            "local_valid_second_bert_081.csv",
        ],
        note="Классификатор надёжности salary hints.",
    ),
    Stage(
        "15_finalize_hint_classifier_081",
        "14_finalize_hint_classifier_081.py",
        outputs=["local_valid_final_boost_081.csv", "submission_081_final_boost.csv", "final_boost_081_results.json"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_hint_classifier_081.csv",
            "test_hint_classifier_081.csv",
            "submission_081_two_bert.csv",
            "hint_classifier_081_results.json",
        ],
        note="Финальный boost 081 с company/location коррекциями.",
    ),
    Stage(
        "16_tfidf_salary_levels_083_local",
        "15_tfidf_salary_levels_083_local.py",
        outputs=["local_valid_tfidf_levels_083.csv", "tfidf_levels_083_results.json"],
        required_inputs=["train.csv", "test.csv"],
        note="Локальные TF-IDF salary-level сигналы.",
    ),
    Stage(
        "17_salary_candidates_083_local",
        "16_salary_candidates_083_local.py",
        outputs=["local_valid_salary_candidate_scores_083.csv", "local_valid_broad_salary_candidates_083.csv", "broad_salary_candidates_083_results.json"],
        required_inputs=["train.csv", "test.csv", "local_valid_final_boost_081.csv"],
        note="Локальный salary-candidate classifier.",
    ),
    Stage(
        "18_rubert_large_base_all",
        "17_rubert_large_base_all.py",
        outputs=["local_valid_rubert_large_081.csv", "test_rubert_large_081.csv", "rubert_large_081_results.json"],
        required_inputs=["train.csv", "test.csv", "rubert_large_base", "local_valid_final_boost_081.csv"],
        note="Fine-tune RuBERT-large.",
        heavy=True,
    ),
    Stage(
        "19_rubert_large_seed2_all",
        "18_rubert_large_seed2_all.py",
        outputs=["local_valid_rubert_large_081_seed2.csv", "test_rubert_large_081_seed2.csv", "rubert_large_081_results_seed2.json"],
        required_inputs=["train.csv", "test.csv", "rubert_large_base", "local_valid_final_boost_081.csv"],
        note="Fine-tune второй seed RuBERT-large.",
        heavy=True,
    ),
    Stage(
        "20_ruroberta_large_local",
        "19_ruroberta_large_local.py",
        outputs=["local_valid_rubert_large_081_roberta.csv", "rubert_large_081_results_roberta.json"],
        required_inputs=["train.csv", "ruroberta_large_base", "local_valid_final_boost_081.csv"],
        note="RuRoBERTa local OOF.",
        heavy=True,
    ),
    Stage(
        "21_xlm_roberta_large_all",
        "20_xlm_roberta_large_all.py",
        outputs=["local_valid_rubert_large_081_xlm.csv", "test_rubert_large_081_xlm.csv", "rubert_large_081_results_xlm.json"],
        required_inputs=["train.csv", "test.csv", "xlm_roberta_large_base", "local_valid_final_boost_081.csv"],
        note="XLM-RoBERTa-large fine-tuning.",
        heavy=True,
    ),
    Stage(
        "22_train_full_tfidf_levels_087",
        "21_train_full_tfidf_levels_087.py",
        outputs=["test_tfidf_levels_087.csv", "full_tfidf_levels_087_results.json"],
        required_inputs=["train.csv", "test.csv", "local_valid_tfidf_levels_083.csv"],
        note="Full-train TF-IDF salary-level scoring.",
    ),
    Stage(
        "23_train_full_salary_candidates_087",
        "22_train_full_salary_candidates_087.py",
        outputs=["test_salary_candidate_scores_087.csv", "full_salary_candidates_087_results.json"],
        required_inputs=["train.csv", "test.csv"],
        note="Full-train salary candidate classifier.",
    ),
    Stage(
        "24_final_candidate_reranker_087",
        "23_final_candidate_reranker_087.py",
        outputs=["submission_candidate_reranker.csv", "submission.csv", "candidate_reranker_results.json"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "sample_submition.csv",
            "local_valid_final_boost_081.csv",
            "local_valid_079_tfidf_te_predictions.csv",
            "local_valid_tfidf_levels_083.csv",
            "local_valid_rubert_large_081.csv",
            "local_valid_rubert_large_081_seed2.csv",
            "local_valid_rubert_large_081_roberta.csv",
            "local_valid_rubert_large_081_xlm.csv",
            "submission_081_final_boost.csv",
            "test_rubert_large_081.csv",
            "test_rubert_large_081_seed2.csv",
            "test_rubert_large_081_xlm.csv",
            "submission_honest_raw_salary_hints.csv",
            "submission_079_tfidf_te_blend_uncalibrated.csv",
            "test_tfidf_levels_087.csv",
            "local_valid_salary_candidate_scores_083.csv",
            "test_salary_candidate_scores_087.csv",
        ],
        note="Финальный Ridge stack + LightGBM/CatBoost reranker.",
    ),
]
