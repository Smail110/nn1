"""Конфигурация полного pipeline.

Все пути считаются относительно корня сдаваемого проекта
``leaderboard_087_full_pipeline``. В этой папке лежат исходные ``train.csv`` /
``test.csv``, промежуточные артефакты и код для воспроизведения финального
submission.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
STEPS_DIR = PIPELINE_DIR / "steps"
STAGE_SCRIPT_DIR = PIPELINE_DIR / "stage_scripts"
PROJECT_DIR = ROOT
SOURCE_ROOT = STEPS_DIR

PYTHON = Path(sys.executable)

LOG_DIR = ROOT / "logs"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
SUBMISSION_DIR = ROOT / "submissions"

RAW_DATA_FILES = [
    "train.csv",
    "test.csv",
    "sample_submition.csv",
]

# Открытые pretrained-модели. Это не ответы соревнования и не внешняя разметка:
# они используются только как языковая база, которую pipeline дообучает на train.
OPEN_PRETRAINED_MODEL_DIRS = [
    "rubert_large_base",
    "ruroberta_large_base",
    "xlm_roberta_large_base",
]

# База DeepPavlov RuBERT из стандартного Hugging Face cache.
DEEPPAVLOV_RUBERT_BASE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--DeepPavlov--rubert-base-cased"
    / "snapshots"
    / "4036cab694767a299f2b9e6492909664d9414229"
)
