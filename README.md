# TETA NN1 2026 — полный pipeline решения

Автор: Роев Герман Александрович  
Kaggle nickname: `Smail110`

Итоговый результат выбранного сабмита:

- Public: `0.814473`
- Private: `0.826875`

Эта папка содержит код и ноутбук для воспроизведения решения соревнования `teta-nn-1-2026`.

## Главные файлы

- `TETA_NN1_FINAL_HOMEWORK.ipynb` — итоговый ноутбук для сдачи ДЗ с сохранёнными outputs.
- `run_all.py` — основной orchestrator полного pipeline.
- `stages.py` — список этапов обучения и финальной сборки.
- `pipeline_utils.py`, `config.py` — общие утилиты и пути.
- `stage_scripts/` — отдельные stage-обёртки и финальный сборщик submission.
- `requirements.txt` — зависимости.
- `submission.csv` — финальный файл для загрузки на Kaggle, создаётся pipeline.

## Как запустить с нуля локально

Из этой папки:

```powershell
cd D:\PythonProject1\leaderboard_087_full_pipeline
..\.venv\Scripts\python.exe -u run_all.py --from-scratch --include-heavy
```

Быстрый resume без пересчёта готовых этапов:

```powershell
..\.venv\Scripts\python.exe -u run_all.py --resume
```

После успешного полного запуска создаются:

- `submission.csv`
- `submission_candidate_reranker.csv`
- `candidate_reranker_results.json`

## Какие данные использует решение

Pipeline использует только:

- `train.csv`
- `test.csv`
- `sample_submition.csv`
- открытые pretrained-модели RuBERT / RuRoBERTa / XLM-RoBERTa как основу для fine-tuning.

Старые Kaggle-сабмиты и ответы leaderboard не используются как входные данные.

## Важное для GitHub

В репозиторий лучше загружать код, ноутбук, README и небольшие JSON-отчёты.  
Не нужно коммитить большие обученные модели, `.npy` embeddings, AutoGluon-папки, логи и CSV-артефакты полного прогона — они создаются локально при запуске.

Если проверяющий запускает проект на другой машине, нужно положить рядом исходные CSV соревнования и доступные pretrained-модели:

- `train.csv`
- `test.csv`
- `sample_submition.csv`
- `rubert_large_base/`
- `ruroberta_large_base/`
- `xlm_roberta_large_base/`

## Что осталось сделать вручную перед сдачей

В ноутбуке уже внесены ФИО, ник и scores. Остаётся:

1. вписать итоговую позицию на leaderboard после финального freeze;
2. добавить screenshot leaderboard в файл `leaderboard_final.png`;
3. убедиться, что screenshot отображается в ноутбуке;
4. загрузить notebook/репозиторий и отправить открытую ссылку.
