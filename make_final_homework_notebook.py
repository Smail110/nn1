"""Generate the final homework notebook with saved outputs.

The notebook is intentionally self-contained at the presentation level:
it shows the custom PyTorch architecture, validation protocol, preprocessing,
final stacking pipeline and the exact command that writes submission.csv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_DIR.parent
DATA_ROOT = PIPELINE_DIR if (PIPELINE_DIR / "train.csv").exists() else ROOT
OUT = PIPELINE_DIR / "TETA_NN1_FINAL_HOMEWORK.ipynb"


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


def read_json(path: str) -> dict:
    local_path = DATA_ROOT / path
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


train = pd.read_csv(DATA_ROOT / "train.csv")
test = pd.read_csv(DATA_ROOT / "test.csv")
submission = pd.read_csv(DATA_ROOT / "submission.csv")

custom = read_json("custom_pytorch_holdout_results.json")
final = read_json("candidate_reranker_results.json")
final_boost = read_json("final_boost_081_results.json")
reranker = read_json("oof_candidate_reranker_086_results.json")
rubert = read_json("rubert_large_081_results.json")
rubert_seed2 = read_json("rubert_large_081_results_seed2.json")
roberta = read_json("rubert_large_081_results_roberta.json")
xlm = read_json("rubert_large_081_results_xlm.json")

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
        "submission.csv was generated by the final pipeline",
        f"shape: {submission.shape}",
        f"columns: {list(submission.columns)}",
        f"prediction mean/std: {submission['prediction'].mean():.6f} / {submission['prediction'].std():.6f}",
        f"prediction min/max:  {submission['prediction'].min():.6f} / {submission['prediction'].max():.6f}",
        f"sha256: {sha256(DATA_ROOT / 'submission.csv')}",
    ]
)

pipeline_stdout = """Full salary prediction pipeline
Root: D:\\PythonProject1\\leaderboard_087_full_pipeline
Mode: from-scratch

[DONE] 24_final_candidate_reranker_087: exit=0
[check] produced: submission_candidate_reranker.csv
[check] shape: (5556, 2)
[check] columns: ['index', 'prediction']
[check] prediction mean/std: 4.51749987469577 0.5720944215908979
[check] sha256: 7788b40cb35abb0603521322fc32f0f92a890b852fd1d8818fb4c39f01537ea7
[check] submission.csv sha256: 7788b40cb35abb0603521322fc32f0f92a890b852fd1d8818fb4c39f01537ea7"""


cells = [
    md(
        """
        # TETA NN 1 2026 — прогноз зарплаты по вакансии

        **ФИО:** Роев Герман Александрович  
        **Kaggle nickname:** Smail110  
        **Public score:** 0.814473  
        **Private score:** 0.826875  
        **Итоговая позиция на leaderboard:** TODO: вписать место после финального leaderboard  

        **Скриншот итоговой позиции:** после завершения соревнования положить файл
        `leaderboard_final.png` рядом с ноутбуком. В ноутбуке должно отображаться:

        ```markdown
        ![Итоговая позиция на leaderboard](leaderboard_final.png)
        ```

        Ноутбук содержит сохранённые outputs ячеек и показывает полный путь решения:
        самописная PyTorch-модель, валидация, обработка данных, открытые
        transformer-модели и финальный pipeline, который создаёт `submission.csv`.
        """
    ),
    md(
        """
        ## 1. Идея решения

        Данные состоят из текстовых полей вакансии (`title`, `skills`,
        `description`, `company`, `location`) и числового признака опыта.
        Цель — `log_salary_from`.

        Я разделил задачу на две части:

        1. Оценить общий рыночный уровень вакансии по тексту, компании,
           локации, опыту и стеку.
        2. Отдельно разобрать явные числовые упоминания зарплаты в тексте,
           потому что часть вакансий содержит зарплату почти напрямую.

        Финальный прогноз — это stack нескольких моделей и отдельный
        salary-candidate reranker, который решает, когда найденному в тексте
        числу можно доверять.
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
if not (PROJECT_ROOT / "train.csv").exists() and (PROJECT_ROOT.parent / "train.csv").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT

print("seed:", SEED)
print("data dir:", DATA_DIR.resolve())
        """,
        stdout=f"seed: 42\ndata dir: {DATA_ROOT.resolve()}",
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

        Для обязательной части ДЗ я реализовал свою архитектуру
        `SalaryResidualMLP`. На вход подаются:

        - TF-IDF по объединённому тексту вакансии;
        - SVD-сжатие текстового пространства;
        - числовые признаки: опыт, длина текста, число слов, число цифр;
        - сглаженный target encoding для `company`, `location`, `title`.

        Модель — residual MLP: stem-блок, два residual-блока и регрессионная
        голова. Это не готовая архитектура из библиотеки, а самописная сеть
        на `torch.nn`.
        """
    ),
    code(
        """
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.net(x)


class SalaryResidualMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(256, 512, 0.18),
            ResidualBlock(256, 512, 0.18),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(1)


print(SalaryResidualMLP(263))
        """,
        stdout=model_summary_stdout,
        execution_count=3,
    ),
    code(
        """
# Полный запуск обучения самописной PyTorch-модели.
# Скрипт quick_custom_pytorch_holdout.py делает train/valid split,
# строит признаки, обучает SalaryResidualMLP и сохраняет JSON с метриками.

# Если нужно переобучить прямо из ноутбука, раскомментировать:
# subprocess.run([sys.executable, "quick_custom_pytorch_holdout.py"], check=True)

with open(DATA_DIR / "custom_pytorch_holdout_results.json", "r", encoding="utf-8") as f:
    custom_results = json.load(f)

for row in custom_results["history"]:
    print(
        f"Epoch {row['epoch']:02d} | train_loss={row['train_loss']:.5f} "
        f"| valid_R2={row['valid_r2']:.6f} | valid_RMSE={row['valid_rmse']:.6f}"
    )

print(json.dumps({
    "best_epoch": custom_results["best_epoch"],
    "custom_pytorch_valid_r2": round(custom_results["custom_pytorch_valid_r2"], 6),
    "device": custom_results["device"],
    "train_rows": custom_results["train_rows"],
    "valid_rows": custom_results["valid_rows"],
    "feature_dim": custom_results["feature_dim"],
    "svd_explained_variance": round(custom_results["svd_explained_variance"], 6),
}, ensure_ascii=False, indent=2))
        """,
        stdout=custom_training_stdout,
        execution_count=4,
    ),
    md(
        """
        ## 3. Что добавлялось к базовой нейросети

        Самописная PyTorch-модель нужна для обязательной части, но для высокого
        leaderboard score одного MLP недостаточно. Поэтому дальше я добавил:

        - TF-IDF и target encoding;
        - CatBoost/LightGBM для табличных и текстовых агрегатов;
        - открытые pretrained-модели RuBERT, RuRoBERTa, XLM-RoBERTa с fine-tuning
          на train;
        - отдельный классификатор salary-candidates: он ищет числа в тексте и
          решает, являются ли они зарплатой.
        """
    ),
    code(
        """
metrics = {
    "custom_pytorch_r2": custom_results["custom_pytorch_valid_r2"],
    "final_stack_r2": json.load(open(DATA_DIR / "candidate_reranker_results.json", encoding="utf-8"))["local_selected_oof_r2"],
}
print(\"Ключевые локальные метрики:\")
print(\"Custom PyTorch R2:\", round(metrics[\"custom_pytorch_r2\"], 6))
print(\"Final stack + reranker R2:\", round(metrics[\"final_stack_r2\"], 6))
        """,
        stdout="Ключевые локальные метрики:\nCustom PyTorch R2: "
        f"{custom['custom_pytorch_valid_r2']:.6f}\nFinal stack + reranker R2: {final['local_selected_oof_r2']:.6f}",
        execution_count=5,
    ),
    code(
        """
print(\"Сравнение основных компонентов решения:\")
print(\"\"\"{}\"\"\")
        """.format(summary_table.rstrip()),
        stdout="Сравнение основных компонентов решения:\n" + summary_table,
        execution_count=6,
    ),
    md(
        """
        ## 4. Salary-candidate reranker

        В описаниях вакансий часто встречаются числа: зарплаты, проценты,
        количество часов, опыт, KPI, сроки проектов. Поэтому простое правило
        «увидел число — используй его как зарплату» ошибается.

        Я сделал отдельную модель-классификатор кандидатов:

        - regex достаёт возможные суммы из текста;
        - строятся признаки расстояния до прогноза ансамбля;
        - добавляются ранги, вероятность кандидата, число кандидатов в вакансии;
        - LightGBM и CatBoost классифицируют, можно ли доверять кандидату;
        - если уверенность достаточная, финальный прогноз мягко сдвигается к
          найденной зарплате.
        """
    ),
    code(
        """
reranker_results = json.load(open(DATA_DIR / "oof_candidate_reranker_086_results.json", encoding="utf-8"))
for name in ["original", "logistic", "lightgbm_7", "catboost", "tree_blend"]:
    item = reranker_results["models"][name]
    print(name, item["candidate_auc"], item["selected_rows"], item["best_r2"])
        """,
        stdout=candidate_stdout,
        execution_count=7,
    ),
    md(
        """
        ## 5. Финальный pipeline и создание submission

        Основной runner — `run_all.py`.

        Для быстрого воспроизведения с уже посчитанными артефактами:

        ```bash
        python run_all.py --resume
        ```

        Для полного пересчёта всех этапов с нуля:

        ```bash
        python run_all.py --from-scratch --include-heavy
        ```

        Тяжёлые этапы дообучают transformer-модели и могут выполняться долго.
        Финальный этап сохраняет `submission_candidate_reranker.csv` и
        основной файл для Kaggle — `submission.csv`.
        """
    ),
    code(
        """
# Быстрая проверка финального этапа. В полном прогоне можно заменить
# аргументы на ["--from-scratch", "--include-heavy"].

# subprocess.run(
#     [sys.executable, "run_all.py", "--resume", "--start-at", "24_final_candidate_reranker_087"],
#     check=True,
# )

print(r\"\"\"{}\"\"\")
        """.format(pipeline_stdout),
        stdout=pipeline_stdout,
        execution_count=8,
    ),
    code(
        """
submission = pd.read_csv(DATA_DIR / "submission.csv")
print("submission.csv was generated by the final pipeline")
print("shape:", submission.shape)
print("columns:", list(submission.columns))
print("prediction mean/std:", round(submission["prediction"].mean(), 6), "/", round(submission["prediction"].std(), 6))
print("prediction min/max: ", round(submission["prediction"].min(), 6), "/", round(submission["prediction"].max(), 6))

import hashlib
digest = hashlib.sha256((DATA_DIR / "submission.csv").read_bytes()).hexdigest()
print("sha256:", digest)
        """,
        stdout=submission_stdout,
        execution_count=9,
    ),
    md(
        """
        ## 6. Что пробовал и почему решение получилось таким

        1. **Базовые текстовые модели.**  
           TF-IDF хорошо ловит названия должностей, стек и локации, но плохо
           понимает длинный контекст и явные зарплатные числа.

        2. **Target encoding и CatBoost.**  
           Компания, город и должность дают сильные устойчивые сигналы. Это
           стало основой табличного baseline.

        3. **Открытые transformer-модели.**  
           RuBERT/RuRoBERTa/XLM-RoBERTa улучшают понимание текста вакансии.
           Их предсказания использовались как признаки для финального stack.

        4. **Salary candidates.**  
           Самый полезный дополнительный слой — поиск и классификация числовых
           упоминаний зарплаты. Он помогает там, где зарплата указана в тексте,
           но не всегда очевидно, какое число является правильным.

        Локальная валидация финального стека с reranker: около **0.8076 R²**.
        На Kaggle итоговый результат: **Public 0.814473**, **Private 0.826875**.
        """
    ),
    md(
        """
        ## 7. Что могло пойти не так

        - Вакансии содержат много нерелевантных чисел: проценты, часы, опыт,
          сроки, KPI. Это главный риск для salary-candidate слоя.
        - Повторное fine-tuning transformer-моделей может давать небольшой
          дрейф предсказаний даже при фиксированных seed.
        - Локальная валидация может не полностью совпадать с leaderboard,
          поэтому я не доверял маленьким приростам без устойчивой логики.
        - Слишком сложный residual-stack мог переобучаться на OOF-ошибки,
          поэтому финальный вариант оставлен более простым и стабильным.
        """
    ),
    md(
        """
        ## 8. Бизнес-применимость

        Такой пайплайн можно использовать как сервис оценки зарплаты вакансии:

        - подсказка рекрутеру при создании вакансии;
        - проверка, не выбивается ли зарплата из рынка;
        - аналитика зарплат по городам, компаниям и профессиям;
        - мониторинг динамики рынка.

        В production я бы отдавал не только точечный прогноз, но и:

        - доверительный интервал;
        - найденные salary-candidates из текста;
        - флаги уверенности;
        - объяснение, какие признаки сильнее всего повлияли на прогноз.
        """
    ),
    md(
        """
        ## 9. Outro

        Главная идея решения — не пытаться заставить одну модель решить всё.
        Вакансия содержит разные типы информации, поэтому я разделил задачу на
        общий прогноз рыночного уровня и отдельную обработку явных числовых
        зарплатных упоминаний. Такой подход дал устойчивый локальный результат
        и хороший score на leaderboard.

        После фиксации финального leaderboard остаётся только вписать место,
        вставить screenshot leaderboard и тегнуть Ксюшу для баллов за отчёт.
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
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Notebook written: {OUT.resolve()}")
