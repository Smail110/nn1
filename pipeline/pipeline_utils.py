"""Общие утилиты для orchestration-части pipeline.

Здесь нет ML-логики: только запуск этапов, проверка файлов, хвосты логов и
финальная валидация submission.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .config import ARTIFACT_DIR, LOG_DIR, REPORT_DIR, ROOT, STAGE_SCRIPT_DIR, STEPS_DIR, SUBMISSION_DIR
except ImportError:  # когда stage-wrapper запускается как обычный .py файл
    from config import ARTIFACT_DIR, LOG_DIR, REPORT_DIR, ROOT, STAGE_SCRIPT_DIR, STEPS_DIR, SUBMISSION_DIR


def rel(path: Path) -> str:
    """Возвращает путь относительно корня проекта, если это возможно."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    """Создаёт служебные папки для логов, отчётов, артефактов и submissions."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    """Считает SHA256 файла."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing(paths: Iterable[Path]) -> list[Path]:
    """Возвращает отсутствующие файлы или папки."""

    return [path for path in paths if not path.exists()]


def read_tail(path: Path, lines: int = 60) -> str:
    """Читает хвост лог-файла для понятного сообщения об ошибке."""

    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def run_parent_script(script_name: str, *args: str) -> None:
    """Запускает реальный скрипт этапа из новой структуры проекта.

    Wrapper'ы лежат в ``pipeline/stage_scripts``, а основная логика этапов —
    в ``pipeline/steps``. Рабочая директория всегда остаётся корнем проекта,
    чтобы все промежуточные CSV/JSON создавались там же, где ожидает pipeline.
    """

    raw_path = Path(script_name)
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [
            STEPS_DIR / raw_path,
            STAGE_SCRIPT_DIR / raw_path,
            ROOT / raw_path,
        ]

    script_path = next((path for path in candidates if path.exists()), None)
    if script_path is None:
        tried = ", ".join(str(path) for path in candidates)
        raise SystemExit(f"Не найден stage-скрипт. Проверенные пути: {tried}")

    command = [sys.executable, "-u", str(script_path), *args]
    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print("[stage] запуск:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(completed.returncode)


def copy_submission_artifacts() -> None:
    """Кладёт финальные CSV в аккуратную папку ``submissions`` для GitHub."""

    for name in ["submission.csv", "submission_candidate_reranker.csv", "submission_final_for_upload.csv"]:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, SUBMISSION_DIR / name)


def validate_final_submission(produced_name: str = "submission_candidate_reranker.csv") -> None:
    """Проверяет финальный submission: формат, NaN, статистику и hash."""

    produced = ROOT / produced_name
    sample = ROOT / "sample_submition.csv"

    if not produced.exists():
        raise SystemExit(f"Финальный submission не найден: {produced}")

    frame = pd.read_csv(produced)
    print("[check] produced:", produced.name)
    print("[check] shape:", frame.shape)
    print("[check] columns:", list(frame.columns))

    if sample.exists():
        sample_frame = pd.read_csv(sample)
        if len(frame) != len(sample_frame) or list(frame.columns) != list(sample_frame.columns):
            raise SystemExit("Формат submission не совпадает с sample_submition.csv")

    if frame["prediction"].isna().any() or not np.isfinite(frame["prediction"]).all():
        raise SystemExit("В submission есть NaN или бесконечные значения")

    print("[check] prediction mean/std:", float(frame["prediction"].mean()), float(frame["prediction"].std()))
    print("[check] sha256:", sha256(produced))

    upload = ROOT / "submission.csv"
    if upload.exists():
        print("[check] submission.csv sha256:", sha256(upload))

    copy_submission_artifacts()
