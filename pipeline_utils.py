"""Общие утилиты для полного pipeline.

В этом файле нет ML-логики: только безопасный запуск стадий, проверки файлов
и финальная валидация submission.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, LOG_DIR, ROOT, SOURCE_ROOT


def rel(path: Path) -> str:
    """Красиво печатает путь относительно корня проекта, если это возможно."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    """Создаёт служебные папки для логов и артефактов."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    """Считает SHA256 файла, чтобы удобно сравнивать submissions."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing(paths: Iterable[Path]) -> list[Path]:
    """Возвращает список отсутствующих файлов/папок."""

    return [path for path in paths if not path.exists()]


def read_tail(path: Path, lines: int = 60) -> str:
    """Читает хвост лог-файла для понятного сообщения об ошибке."""

    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def run_parent_script(script_name: str, *args: str) -> None:
    """Запускает исходный stage-скрипт из корня проекта.

    Stage-wrapper'ы в `stage_scripts/` вызывают эту функцию. Так мы держим
    отдельные профессиональные файлы стадий, но не дублируем огромный код
    исходных экспериментов.
    """

    script_path = ROOT / script_name
    if not script_path.exists():
        script_path = SOURCE_ROOT / script_name
    if not script_path.exists():
        raise SystemExit(f"Не найден stage-скрипт: {script_path}")

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

def validate_final_submission(
    produced_name: str = "submission_candidate_reranker.csv",
) -> None:
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
