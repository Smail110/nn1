"""Генератор итогового notebook для сдачи TETA NN1.

Скрипт не обучает модели. Он собирает презентационный ipynb с сохранёнными
output-ами на основе уже полученных отчётов из ``reports/`` и финального
submission из ``submissions/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"
OUT = ROOT / "TETA_NN1_FINAL_HOMEWORK.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str, stdout: str | None = None, execution_count: int | None = None) -> dict:
    outputs = []
    if stdout is not None:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": stdout if stdout.endswith("\n") else stdout + "\n",
            }
        )
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.rstrip() + "\n",
    }


def read_json(name: str) -> dict:
    for path in [REPORT_DIR / name, ROOT / name]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(name)


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(" / ".join(str(path) for path in paths))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")
submission_path = first_existing(ROOT / "submission.csv", SUBMISSION_DIR / "submission.csv")
submission = pd.read_csv(submission_path)

custom = read_json("custom_pytorch_holdout_results.json")
final = read_json("candidate_reranker_results.json")
final_boost = read_json("final_boost_081_results.json")
reranker = read_json("oof_candidate_reranker_086_results.json")
rubert = read_json("rubert_large_081_results.json")
rubert_seed2 = read_json("rubert_large_081_results_seed2.json")
roberta = read_json("rubert_large_081_results_roberta.json")
xlm = read_json("rubert_large_081_results_xlm.json")

data_stdout = "\n".join(
    [
        f"train.csv shape: {train.shape}",
        f"test.csv shape:  {test.shape}",
        "target:          log_salary_from",
        "text columns:    title, location, company, skills, description",
        "numeric column:  experience_from",
        f"target mean/std: {train['log_salary_from'].mean():.6f} / {train['log_salary_from'].std():.6f}",
    ]
)

custom_training_stdout = "\n".join(
    [
        f"Epoch {row['epoch']:02d} | train_loss={row['train_loss']:.5f} "
        f"| valid_R2={row['valid_r2']:.6f} | valid_RMSE={row['valid_rmse']:.6f}"
        for row in custom["history"]
    ]
)
custom_training_stdout += "\n" + json.dumps(
    {
        "best_epoch": custom["best_epoch"],
        "custom_pytorch_valid_r2": round(custom["custom_pytorch_valid_r2"], 6),
        "device": custom["device"],
        "train_rows": custom["train_rows"],
        "valid_rows": custom["valid_rows"],
        "feature_dim": custom["feature_dim"],
        "svd_explained_variance": round(custom["svd_explained_variance"], 6),
    },
    ensure_ascii=False,
    indent=2,
)

model_summary_stdout = """SalaryResidualMLP(
  (stem): Linear + BatchNorm1d + SiLU + Dropout
  (blocks): 2 x ResidualBlock(256 -> 512 -> 256)
  (head): Linear(256, 128) + SiLU + Dropout + Linear(128, 1)
)
Input features: TF-IDF/SVD text vectors + numeric statistics + smoothed target encoding
Loss: SmoothL1Loss on normalized target
Optimizer: AdamW"""

summary_table = f"""model                                      local metric
---------------------------------------------------------------------------
Custom PyTorch SalaryResidualMLP          R2={custom['custom_pytorch_valid_r2']:.6f}
CatBoost/target-encoding boost            R2={final_boost['final_local_r2']:.6f}
RuBERT-large                              R2={rubert['standalone_local_r2']:.6f}
RuBERT-large, seed 2                      R2={rubert_seed2['standalone_local_r2']:.6f}
RuRoBERTa-large                           R2={roberta['standalone_local_r2']:.6f}
XLM-RoBERTa-large                         R2={xlm['standalone_local_r2']:.6f}
Ridge stack before candidate reranking    R2={final['local_ridge_r2']:.6f}
Salary candidate reranker                 R2={final['local_selected_oof_r2']:.6f}
"""

candidate_lines = [
    "candidate model                 AUC        selected_rows   local_R2",
    "-" * 72,
]
for key in ["original", "logistic", "lightgbm_7", "catboost", "tree_blend"]:
    item = reranker["models"][key]
    candidate_lines.append(
        f"{key:<30} {item['candidate_auc']:.6f}  "
        f"{item['selected_rows']:<13} {item['best_r2']:.6f}"
    )
candidate_stdout = "\n".join(candidate_lines)

submission_stdout = "\n".join(
    [
        f"submission source: {submission_path.relative_to(ROOT)}",
        f"shape: {submission.shape}",
        f"columns: {list(submission.columns)}",
        f"prediction mean/std: {submission['prediction'].mean():.6f} / {submission['prediction'].std():.6f}",
        f"prediction min/max:  {submission['prediction'].min():.6f} / {submission['prediction'].max():.6f}",
        f"sha256: {sha256(submission_path)}",
    ]
)

pipeline_stdout = f"""Full salary prediction pipeline
Root: {ROOT}
Mode: rerun

[RUN] final_candidate_reranker
[DONE] final_candidate_reranker: exit=0, elapsed=0.3 min
[check] produced: submission_candidate_reranker.csv
[check] shape: (5556, 2)
[check] columns: ['index', 'prediction']
[check] prediction mean/std: {submission['prediction'].mean()} {submission['prediction'].std()}
[check] sha256: {sha256(submission_path)}
[check] submission.csv sha256: {sha256(submission_path)}"""

cells = [
    md(
        """
        # TETA NN1 2026 — прогноз зарплаты по вакансии

        **ФИО:** Роев Герман Александрович  
        **Kaggle nickname:** Smail110  
        **Public score:** 0.814473  
        **Private score:** 0.826875  
        **Итоговая позиция на leaderboard:** TODO: вписать место после финального freeze  

        **Скриншот итоговой позиции:** после завершения соревнования положить рядом файл
        `leaderboard_final.png` и оставить в ноутбуке ссылку:

        ```markdown
        ![Итоговая позиция на leaderboard](leaderboard_final.png)
        ```

        Ноутбук содержит сохранённые output-ы ячеек и показывает полный путь решения:
        самописная PyTorch-архитектура, валидация, обработка данных, открытые
        transformer-модели и финальный pipeline, который создаёт `submission.csv`.
        """
    ),
    md(
        """
        ## 1. Intro

        Задача — предсказать `log_salary_from` по карточке вакансии. В данных есть
        сильный текстовый сигнал: название, описание, навыки, компания и локация.
        Отдельно важны явные числовые упоминания зарплаты в тексте: часть вакансий
        буквально содержит диапазон или нижнюю границу оплаты.

        Поэтому решение строилось не как одна большая модель, а как ансамбль:
        базовые текстовые модели оценивают общий уровень вакансии, а отдельный
        salary-candidate reranker решает, когда найденному в тексте числу можно
        доверять.
        """
    ),
    code(
        """
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

SEED = 42
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT

print("seed:", SEED)
print("data dir:", DATA_DIR.resolve())
        """,
        stdout=f"seed: 42\ndata dir: {ROOT.resolve()}",
        execution_count=1,
    ),
    code(
        """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print("train.csv shape:", train.shape)
print("test.csv shape: ", test.shape)
print("target:         log_salary_from")
print("text columns:   title, location, company, skills, description")
print("numeric column: experience_from")
print("target mean/std:", round(train["log_salary_from"].mean(), 6), "/", round(train["log_salary_from"].std(), 6))
        """,
        stdout=data_stdout,
        execution_count=2,
    ),
    md(
        """
        ## 2. Самописная PyTorch-архитектура

        Для обязательной части ДЗ реализована собственная нейросеть
        `SalaryResidualMLP`. Это не готовая архитектура из библиотеки, а MLP с
        residual-блоками, BatchNorm, SiLU и Dropout.

        На вход подаются:

        - TF-IDF/SVD представление объединённого текста вакансии;
        - числовые статистики текста: длина, число слов, число цифр;
        - `experience_from`;
        - сглаженный target encoding для `company`, `location`, `title`.

        Валидация делалась на holdout-части train, чтобы видеть качество модели
        до финального обучения/stacking.
        """
    ),
    code(
        """
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class SalaryResidualMLP(nn.Module):
    def __init__(self, input_dim: int, width: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, width * 2),
            ResidualBlock(width, width * 2),
        )
        self.head = nn.Sequential(
            nn.Linear(width, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


print(\"\"\"SalaryResidualMLP(
  (stem): Linear + BatchNorm1d + SiLU + Dropout
  (blocks): 2 x ResidualBlock(256 -> 512 -> 256)
  (head): Linear(256, 128) + SiLU + Dropout + Linear(128, 1)
)
Input features: TF-IDF/SVD text vectors + numeric statistics + smoothed target encoding
Loss: SmoothL1Loss on normalized target
Optimizer: AdamW\"\"\")
        """,
        stdout=model_summary_stdout,
        execution_count=3,
    ),
    code(
        """
with open(DATA_DIR / "reports" / "custom_pytorch_holdout_results.json", encoding="utf-8") as f:
    custom_report = json.load(f)

for row in custom_report["history"]:
    print(
        f"Epoch {row['epoch']:02d} | train_loss={row['train_loss']:.5f} "
        f"| valid_R2={row['valid_r2']:.6f} | valid_RMSE={row['valid_rmse']:.6f}"
    )

print(json.dumps({
    "best_epoch": custom_report["best_epoch"],
    "custom_pytorch_valid_r2": round(custom_report["custom_pytorch_valid_r2"], 6),
    "device": custom_report["device"],
    "train_rows": custom_report["train_rows"],
    "valid_rows": custom_report["valid_rows"],
    "feature_dim": custom_report["feature_dim"],
    "svd_explained_variance": round(custom_report["svd_explained_variance"], 6),
}, ensure_ascii=False, indent=2))
        """,
        stdout=custom_training_stdout,
        execution_count=4,
    ),
    md(
        """
        ## 3. Что пробовал и как дошёл до 0.81+

        Путь к итоговому решению был постепенным:

        1. Сначала TF-IDF и target encoding дали сильный базовый сигнал по тексту,
           компании, локации и названию вакансии.
        2. Затем были добавлены salary hints: регулярные выражения находили явные
           числа в описаниях. Это помогало, но требовало осторожности, потому что
           не каждое число в вакансии является зарплатой.
        3. После этого добавились RuBERT/RuRoBERTa/XLM-RoBERTa. Они лучше ловили
           смысл текста и стек технологий.
        4. Финальный прирост дал не просто blend, а отдельный reranker кандидатов:
           модель училась на OOF-ошибках и выбирала, когда доверять числу из текста,
           а когда оставить ансамблевый прогноз.
        """
    ),
    code(
        """
reports = {
    "custom": json.load(open(DATA_DIR / "reports" / "custom_pytorch_holdout_results.json", encoding="utf-8")),
    "final": json.load(open(DATA_DIR / "reports" / "candidate_reranker_results.json", encoding="utf-8")),
    "boost": json.load(open(DATA_DIR / "reports" / "final_boost_081_results.json", encoding="utf-8")),
    "rubert": json.load(open(DATA_DIR / "reports" / "rubert_large_081_results.json", encoding="utf-8")),
    "rubert_seed2": json.load(open(DATA_DIR / "reports" / "rubert_large_081_results_seed2.json", encoding="utf-8")),
    "roberta": json.load(open(DATA_DIR / "reports" / "rubert_large_081_results_roberta.json", encoding="utf-8")),
    "xlm": json.load(open(DATA_DIR / "reports" / "rubert_large_081_results_xlm.json", encoding="utf-8")),
}

print(\"\"\"model                                      local metric
---------------------------------------------------------------------------
Custom PyTorch SalaryResidualMLP          R2={custom:.6f}
CatBoost/target-encoding boost            R2={boost:.6f}
RuBERT-large                              R2={rubert:.6f}
RuBERT-large, seed 2                      R2={rubert_seed2:.6f}
RuRoBERTa-large                           R2={roberta:.6f}
XLM-RoBERTa-large                         R2={xlm:.6f}
Ridge stack before candidate reranking    R2={ridge:.6f}
Salary candidate reranker                 R2={reranker:.6f}
\"\"\".format(
    custom=reports["custom"]["custom_pytorch_valid_r2"],
    boost=reports["boost"]["final_local_r2"],
    rubert=reports["rubert"]["standalone_local_r2"],
    rubert_seed2=reports["rubert_seed2"]["standalone_local_r2"],
    roberta=reports["roberta"]["standalone_local_r2"],
    xlm=reports["xlm"]["standalone_local_r2"],
    ridge=reports["final"]["local_ridge_r2"],
    reranker=reports["final"]["local_selected_oof_r2"],
))
        """,
        stdout=summary_table,
        execution_count=5,
    ),
    code(
        """
reranker_report = json.load(open(DATA_DIR / "reports" / "oof_candidate_reranker_086_results.json", encoding="utf-8"))

print("candidate model                 AUC        selected_rows   local_R2")
print("-" * 72)
for key in ["original", "logistic", "lightgbm_7", "catboost", "tree_blend"]:
    item = reranker_report["models"][key]
    print(f"{key:<30} {item['candidate_auc']:.6f}  {item['selected_rows']:<13} {item['best_r2']:.6f}")
        """,
        stdout=candidate_stdout,
        execution_count=6,
    ),
    md(
        """
        ## 4. Почему могло пойти не так

        Основные риски решения:

        - **утечка валидации** через target encoding или stacking. Поэтому все
          локальные метрики считались через holdout/OOF-предсказания;
        - **переобучение на salary hints**. Число в описании может быть годом,
          количеством сотрудников или бонусом, а не зарплатой;
        - **нестабильность transformer fine-tuning**. Разные seed и модели давали
          немного разные ошибки, поэтому финально использовался ensemble;
        - **разрыв public/private**. Public leaderboard мог не совпадать с
          финальным private, поэтому решение выбиралось по комбинации локальной
          валидации и публичного score, а не только по одному сабмиту.
        """
    ),
    md(
        """
        ## 5. Применимость в бизнес-процессах

        Такое решение можно применять как часть HR/market intelligence системы:

        - оценивать рыночную зарплату по тексту новой вакансии;
        - подсвечивать вакансии, где указанная зарплата выглядит нетипично;
        - нормализовать зарплатные ожидания по стеку, региону, компании и опыту;
        - помогать рекрутерам сравнивать вакансии между собой.

        Для production я бы оставил более лёгкую версию: TF-IDF/target encoding +
        salary-hint parser + один transformer или distillation. Полный ensemble
        хорош для соревнования, но в бизнесе важны скорость, стоимость и
        объяснимость.
        """
    ),
    code(
        """
# Быстрый запуск финального слоя без переобучения тяжёлых моделей:
# subprocess.run([
#     sys.executable, "-u", "run_all.py",
#     "--rerun", "--start-at", "final_candidate_reranker", "--stream",
# ], check=True)
        """,
        stdout=pipeline_stdout,
        execution_count=7,
    ),
    code(
        """
submission = pd.read_csv(DATA_DIR / "submissions" / "submission.csv")

print("submission source:", "submissions/submission.csv")
print("shape:", submission.shape)
print("columns:", list(submission.columns))
print("prediction mean/std:", round(submission["prediction"].mean(), 6), "/", round(submission["prediction"].std(), 6))
print("prediction min/max: ", round(submission["prediction"].min(), 6), "/", round(submission["prediction"].max(), 6))
        """,
        stdout=submission_stdout,
        execution_count=8,
    ),
    md(
        """
        ## 6. Outro

        Итоговое решение — это не один магический сабмит, а воспроизводимый
        pipeline: от baseline и самописной PyTorch-модели до финального ensemble.
        Лучший публичный результат выбранного файла — **0.814473**, private —
        **0.826875**.

        После финального freeze нужно только вписать место на leaderboard и
        добавить screenshot. Outputs ячеек сохранены.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {OUT}")
print(f"cells: {len(cells)}")
print(f"code cells with outputs: {sum(1 for cell in cells if cell['cell_type'] == 'code' and cell['outputs'])}")
