"""Реестр этапов полного pipeline.

Каждый этап описывает один запускаемый wrapper из ``pipeline/stage_scripts``:
какие входные файлы ему нужны, какие артефакты он должен создать и считается ли
он тяжёлым по времени/ресурсам. Имена этапов специально оставлены человекочитаемыми
без технических числовых префиксов — так проще читать логи и README.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .config import ROOT, STAGE_SCRIPT_DIR
except ImportError:
    from config import ROOT, STAGE_SCRIPT_DIR


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
        return [sys.executable, "-u", str(STAGE_SCRIPT_DIR / self.script)]

    def output_paths(self) -> list[Path]:
        return [ROOT / item for item in self.outputs]

    def input_paths(self) -> list[Path]:
        return [ROOT / item for item in self.required_inputs]

    def is_done(self) -> bool:
        paths = self.output_paths()
        return bool(paths) and all(path.exists() for path in paths)


STAGES: list[Stage] = [
    Stage(
        "check_inputs",
        "check_inputs.py",
        outputs=["artifacts/00_inputs_ok.json"],
        required_inputs=["train.csv", "test.csv", "sample_submition.csv"],
        note="Проверка исходных CSV и локальных папок с открытыми pretrained-моделями.",
    ),
    Stage(
        "train_autogluon_text_baseline",
        "train_autogluon_text_baseline.py",
        outputs=["local_valid_raw_text_salary_hints_predictions.csv", "submission_honest_raw_salary_hints.csv"],
        required_inputs=["train.csv", "test.csv"],
        note="AutoGluon/text baseline с признаками из зарплатных подсказок.",
        heavy=True,
    ),
    Stage(
        "train_tfidf_target_encoding",
        "train_tfidf_target_encoding.py",
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
        note="TF-IDF + target encoding как сильный таблично-текстовый baseline.",
    ),
    Stage(
        "extract_salary_hints",
        "extract_salary_hints.py",
        outputs=["local_valid_salary_leak_v2.csv", "submission_079_salary_leak_v2.csv", "test_salary_hints_v2.csv"],
        required_inputs=["train.csv", "test.csv"],
        note="Regex-извлечение явных зарплатных подсказок из текста вакансий.",
    ),
    Stage(
        "blend_knn_text",
        "blend_knn_text.py",
        outputs=["local_valid_knn_079.csv", "submission_079_salary_leak_knn.csv"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_salary_leak_v2.csv",
            "submission_079_salary_leak_v2.csv",
        ],
        note="KNN/text-коррекция поверх salary-hint baseline.",
    ),
    Stage(
        "train_chunked_rubert",
        "train_chunked_rubert.py",
        outputs=["rubert_salary_chunked_full_train"],
        required_inputs=["train.csv", "test.csv"],
        note="Тяжёлая full-train стадия с chunked RuBERT.",
        heavy=True,
    ),
    Stage(
        "blend_chunked_rubert",
        "blend_chunked_rubert.py",
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
        note="Получение local/test prediction-ов chunked RuBERT и blend с TF-IDF.",
        heavy=True,
    ),
    Stage(
        "calibrate_salary_hints",
        "calibrate_salary_hints.py",
        outputs=[
            "local_valid_supervised_salary_hint_081.csv",
            "submission_081_supervised_salary_hint.csv",
            "supervised_salary_hint_081_results.json",
        ],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_honest_bert_079.csv",
            "submission_079_honest_bert_blend.csv",
        ],
        note="Supervised-калибровка явных salary hints.",
    ),
    Stage(
        "train_raw_rubert",
        "train_raw_rubert.py",
        outputs=["test_full_raw_bert_081.csv", "rubert_salary_full_raw_081"],
        required_inputs=["train.csv", "test.csv"],
        note="Full-data RuBERT по сырой целевой переменной.",
        heavy=True,
    ),
    Stage(
        "finalize_raw_rubert",
        "finalize_raw_rubert.py",
        outputs=["submission_081_raw_bert_salary.csv", "submission_081_raw_bert_no_hint.csv", "raw_bert_081_results.json"],
        required_inputs=[
            "local_valid_knn_079.csv",
            "local_valid_honest_bert_079.csv",
            "local_valid_supervised_salary_hint_081.csv",
            "test_full_raw_bert_081.csv",
        ],
        note="Сборка raw-RuBERT submission и локального отчёта.",
    ),
    Stage(
        "train_second_rubert_local",
        "train_second_rubert_local.py",
        outputs=["rubert_salary_local_valid_only"],
        required_inputs=["train.csv"],
        note="Обучение local-valid BERT модели для честной OOF-оценки.",
        heavy=True,
    ),
    Stage(
        "evaluate_second_rubert",
        "evaluate_second_rubert.py",
        outputs=["local_valid_second_bert_081.csv"],
        required_inputs=["train.csv", "rubert_salary_local_valid_only"],
        note="Оценка сохранённой local BERT модели.",
        heavy=True,
    ),
    Stage(
        "train_second_rubert_full",
        "train_second_rubert_full.py",
        outputs=["test_full_second_bert_081.csv", "rubert_salary_full_second_081"],
        required_inputs=["train.csv", "test.csv"],
        note="Второй full-data RuBERT с другим seed/режимом.",
        heavy=True,
    ),
    Stage(
        "finalize_two_rubert",
        "finalize_two_rubert.py",
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
        note="Blend двух RuBERT-моделей.",
    ),
    Stage(
        "train_salary_hint_classifier",
        "train_salary_hint_classifier.py",
        outputs=[
            "local_valid_hint_classifier_081.csv",
            "test_hint_classifier_081.csv",
            "submission_081_hint_classifier.csv",
            "hint_classifier_081_results.json",
        ],
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
        "finalize_hint_classifier",
        "finalize_hint_classifier.py",
        outputs=["local_valid_final_boost_081.csv", "submission_081_final_boost.csv", "final_boost_081_results.json"],
        required_inputs=[
            "train.csv",
            "test.csv",
            "local_valid_hint_classifier_081.csv",
            "test_hint_classifier_081.csv",
            "submission_081_two_bert.csv",
            "hint_classifier_081_results.json",
        ],
        note="Финальный boost с company/location corrections.",
    ),
    Stage(
        "train_tfidf_salary_levels_local",
        "train_tfidf_salary_levels_local.py",
        outputs=["local_valid_tfidf_levels_083.csv", "tfidf_levels_083_results.json"],
        required_inputs=["train.csv", "test.csv"],
        note="Локальные TF-IDF salary-level сигналы.",
    ),
    Stage(
        "train_salary_candidates_local",
        "train_salary_candidates_local.py",
        outputs=[
            "local_valid_salary_candidate_scores_083.csv",
            "local_valid_broad_salary_candidates_083.csv",
            "broad_salary_candidates_083_results.json",
        ],
        required_inputs=["train.csv", "test.csv", "local_valid_final_boost_081.csv"],
        note="Локальный salary-candidate classifier.",
    ),
    Stage(
        "train_large_rubert",
        "train_large_rubert.py",
        outputs=["local_valid_rubert_large_081.csv", "test_rubert_large_081.csv", "rubert_large_081_results.json"],
        required_inputs=["train.csv", "test.csv", "rubert_large_base", "local_valid_final_boost_081.csv"],
        note="Fine-tuning RuBERT-large.",
        heavy=True,
    ),
    Stage(
        "train_large_rubert_seed",
        "train_large_rubert_seed.py",
        outputs=["local_valid_rubert_large_081_seed2.csv", "test_rubert_large_081_seed2.csv", "rubert_large_081_results_seed2.json"],
        required_inputs=["train.csv", "test.csv", "rubert_large_base", "local_valid_final_boost_081.csv"],
        note="Fine-tuning RuBERT-large со вторым seed.",
        heavy=True,
    ),
    Stage(
        "train_large_ruroberta_local",
        "train_large_ruroberta_local.py",
        outputs=["local_valid_rubert_large_081_roberta.csv", "rubert_large_081_results_roberta.json"],
        required_inputs=["train.csv", "ruroberta_large_base", "local_valid_final_boost_081.csv"],
        note="Local OOF для RuRoBERTa-large.",
        heavy=True,
    ),
    Stage(
        "train_large_xlm_roberta",
        "train_large_xlm_roberta.py",
        outputs=["local_valid_rubert_large_081_xlm.csv", "test_rubert_large_081_xlm.csv", "rubert_large_081_results_xlm.json"],
        required_inputs=["train.csv", "test.csv", "xlm_roberta_large_base", "local_valid_final_boost_081.csv"],
        note="Fine-tuning XLM-RoBERTa-large.",
        heavy=True,
    ),
    Stage(
        "train_tfidf_salary_levels_full",
        "train_tfidf_salary_levels_full.py",
        outputs=["test_tfidf_levels_087.csv", "full_tfidf_levels_087_results.json"],
        required_inputs=["train.csv", "test.csv", "local_valid_tfidf_levels_083.csv"],
        note="Full-train TF-IDF salary-level scoring.",
    ),
    Stage(
        "train_salary_candidates_full",
        "train_salary_candidates_full.py",
        outputs=["test_salary_candidate_scores_087.csv", "full_salary_candidates_087_results.json"],
        required_inputs=["train.csv", "test.csv"],
        note="Full-train salary candidate classifier.",
    ),
    Stage(
        "final_candidate_reranker",
        "final_candidate_reranker.py",
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
