"""Конфигурация полного pipeline.

Все пути считаются относительно корня проекта `D:/PythonProject1`.
В этой папке лежит только orchestration-код; исходные данные и тяжёлые
модельные директории остаются в корне проекта.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Папка `leaderboard_087_full_pipeline`.
PIPELINE_DIR = Path(__file__).resolve().parent

# Корень этого сдаваемого pipeline-проекта.
# Здесь лежат train.csv/test.csv и generated artifacts для проверки.
ROOT = PIPELINE_DIR

# Родительский рабочий каталог с исходными экспериментальными скриптами.
# Нужен только как fallback для тяжёлых стадий, чтобы не дублировать большой код.
SOURCE_ROOT = PIPELINE_DIR.parent

# Python текущего виртуального окружения.
PYTHON = Path(sys.executable)

# Служебные папки orchestration-проекта.
LOG_DIR = PIPELINE_DIR / "logs"
ARTIFACT_DIR = PIPELINE_DIR / "artifacts"

# Исходные файлы соревнования.
RAW_DATA_FILES = [
    "train.csv",
    "test.csv",
    "sample_submition.csv",
]

# Открытые pretrained-модели. Они не являются ответами соревнования:
# это базовые языковые модели, от которых выполняется fine-tuning.
OPEN_PRETRAINED_MODEL_DIRS = [
    "rubert_large_base",
    "ruroberta_large_base",
    "xlm_roberta_large_base",
]

# Ещё одна открытая pretrained-база, которую используют BERT-стадии 05/06/08/10/11.
# Она хранится в стандартном Hugging Face cache. Это не обученная на соревновании модель.
DEEPPAVLOV_RUBERT_BASE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--DeepPavlov--rubert-base-cased"
    / "snapshots"
    / "4036cab694767a299f2b9e6492909664d9414229"
)
