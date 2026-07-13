# TETA NN1 2026 — прогноз зарплаты по вакансии

Автор: **Роев Герман Александрович**  
Kaggle nickname: **Smail110**

Итоговый результат выбранного решения:

- Public leaderboard: **0.814473**
- Private leaderboard: **0.826875**

Репозиторий содержит полный код воспроизведения решения для соревнования
`teta-nn-1-2026`: от подготовки признаков и обучения моделей до финального
`submission.csv`.

## Структура проекта

```text
.
├── TETA_NN1_FINAL_HOMEWORK.ipynb   # ноутбук для сдачи ДЗ с сохранёнными output-ами
├── run_all.py                      # главный запуск pipeline
├── pipeline/
│   ├── stages.py                   # реестр этапов
│   ├── config.py                   # пути и настройки
│   ├── pipeline_utils.py           # запуск этапов и проверки
│   ├── stage_scripts/              # лёгкие wrapper-ы этапов
│   └── steps/                      # основная ML-логика этапов
├── reports/                        # сохранённые JSON-отчёты метрик
├── submissions/                    # финальные CSV для Kaggle
├── tools/                          # вспомогательные утилиты
├── requirements.txt
└── FILES_TO_SUBMIT.txt
```

Корень специально оставлен спокойным: тяжёлые модели, логи, промежуточные CSV,
embedding-кэши и AutoGluon-директории создаются локально и не коммитятся.

## Быстрая проверка работоспособности

Если промежуточные prediction-файлы уже лежат в папке проекта, можно быстро
пересобрать финальный submission без переобучения BERT/AutoGluon:

```powershell
cd D:\PythonProject1\leaderboard_087_full_pipeline
..\.venv\Scripts\python.exe -u run_all.py --rerun --start-at final_candidate_reranker --stream
```

Проверенный smoke-test пересобирает:

- `submission_candidate_reranker.csv`
- `submission.csv`
- `submission_final_for_upload.csv`

и дополнительно копирует финальные CSV в папку `submissions/`.

## Полный запуск с нуля

Перед полным запуском в папке проекта должны лежать исходные файлы соревнования:

- `train.csv`
- `test.csv`
- `sample_submition.csv`

Также нужны открытые pretrained-модели/кэши для transformer-этапов:

- `rubert_large_base/`
- `ruroberta_large_base/`
- `xlm_roberta_large_base/`
- Hugging Face cache для `DeepPavlov/rubert-base-cased`

Команда полного запуска:

```powershell
cd D:\PythonProject1\leaderboard_087_full_pipeline
..\.venv\Scripts\python.exe -u run_all.py --from-scratch --include-heavy --stream
```

Полный прогон долгий: он заново обучает AutoGluon, RuBERT, RuRoBERTa,
XLM-RoBERTa, TF-IDF/target-encoding модели и финальный reranker.

## Что использует решение

Pipeline использует только:

- `train.csv`;
- `test.csv`;
- `sample_submition.csv`;
- открытые pretrained language models как базу для fine-tuning.

Старые Kaggle submissions, leaderboard-ответы и внешняя разметка не используются
как входные данные.

## Кратко по модели

В решении собраны несколько групп сигналов:

- самописная PyTorch-модель `SalaryResidualMLP` для обязательной части ДЗ;
- TF-IDF + target encoding baseline;
- regex/ML обработка явных зарплатных упоминаний в тексте;
- несколько fine-tuned transformer-моделей: RuBERT, RuRoBERTa, XLM-RoBERTa;
- финальный Ridge/LightGBM/CatBoost stack и salary-candidate reranker.

Локальная метрика финального reranker:

- Ridge stack до reranking: `R2 = 0.795469`
- итоговый selected OOF: `R2 = 0.807635`

Финальный Kaggle-файл для загрузки лежит в:

```text
submissions/submission.csv
```

## Что добавить вручную перед финальной сдачей

В ноутбуке уже указаны ФИО, ник Kaggle и public/private score. После финального
freeze leaderboard нужно вручную:

1. вписать итоговую позицию;
2. добавить screenshot leaderboard, например `leaderboard_final.png`;
3. убедиться, что output ячеек в `.ipynb` не очищен.
