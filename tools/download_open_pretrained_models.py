"""Опциональная загрузка открытых pretrained-моделей.

Основной pipeline не скачивает модели автоматически, потому что на Kaggle/в локальной
среде интернет часто выключен. Если папок `rubert_large_base`,
`ruroberta_large_base`, `xlm_roberta_large_base` ещё нет, этот скрипт можно
запустить заранее в среде с интернетом.

Важно: это не скачивание чужих ответов соревнования. Это открытые языковые
модели, которые затем дообучаются на `train.csv`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEEPPAVLOV_RUBERT_BASE, ROOT


MODEL_MAP = {
    # Русский BERT-large от AI-Forever/SberDevices.
    "rubert_large_base": "ai-forever/ruBert-large",
    # Русская RoBERTa-large от AI-Forever/SberDevices.
    "ruroberta_large_base": "ai-forever/ruRoberta-large",
    # Мультиязычная XLM-RoBERTa-large от FacebookAI.
    "xlm_roberta_large_base": "FacebookAI/xlm-roberta-large",
}

EXACT_PATH_MODEL_MAP = {
    # Эта база используется несколькими ruBERT-base стадиями через local_files_only=True.
    DEEPPAVLOV_RUBERT_BASE: "DeepPavlov/rubert-base-cased",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Скачать открытые pretrained-модели в локальные папки проекта.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Перекачать даже если папка уже существует")
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Не установлен huggingface_hub. Установи: pip install huggingface_hub"
        ) from exc

    args = parse_args()

    for local_dir, model_id in MODEL_MAP.items():
        target = ROOT / local_dir
        if target.exists() and not args.force:
            print(f"[skip] {local_dir}: папка уже существует")
            continue

        print(f"[download] {model_id} -> {target}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )

    for target, model_id in EXACT_PATH_MODEL_MAP.items():
        if target.exists() and not args.force:
            print(f"[skip] {model_id}: cache already exists at {target}")
            continue

        print(f"[download] {model_id} -> {target}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )

    print("[done] открытые pretrained-модели готовы")


if __name__ == "__main__":
    main()
