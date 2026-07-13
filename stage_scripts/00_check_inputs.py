"""Стадия 00: проверка входных файлов и открытых pretrained-моделей.

Эта стадия не обучает модель. Она фиксирует, что pipeline стартует только от
исходных train/test/sample и от локально сохранённых открытых языковых моделей.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import ARTIFACT_DIR, DEEPPAVLOV_RUBERT_BASE, OPEN_PRETRAINED_MODEL_DIRS, RAW_DATA_FILES, ROOT, SOURCE_ROOT


# Нужен для читаемых русских сообщений в Windows PowerShell.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    report: dict[str, object] = {
        "raw_data_files": {},
        "open_pretrained_model_dirs": {},
        "open_pretrained_model_paths": {},
    }

    missing: list[str] = []

    for name in RAW_DATA_FILES:
        path = ROOT / name
        if not path.exists():
            missing.append(name)
            report["raw_data_files"][name] = {"exists": False}
            continue

        frame = pd.read_csv(path)
        report["raw_data_files"][name] = {
            "exists": True,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    for name in OPEN_PRETRAINED_MODEL_DIRS:
        path = ROOT / name
        if not path.exists():
            path = SOURCE_ROOT / name
        exists = path.exists() and path.is_dir()
        report["open_pretrained_model_dirs"][name] = {
            "exists": exists,
            "path": str(path),
        }
        if not exists:
            missing.append(name)

    deep_pavlov_exists = DEEPPAVLOV_RUBERT_BASE.exists() and DEEPPAVLOV_RUBERT_BASE.is_dir()
    report["open_pretrained_model_paths"]["DeepPavlov/rubert-base-cased"] = {
        "exists": deep_pavlov_exists,
        "path": str(DEEPPAVLOV_RUBERT_BASE),
    }
    if not deep_pavlov_exists:
        missing.append("DeepPavlov/rubert-base-cased cache")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_DIR / "00_inputs_ok.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing:
        print("[inputs] Не хватает обязательных входов:")
        for item in missing:
            print("  -", item)
        raise SystemExit(2)

    print("[inputs] train/test/sample и open pretrained модели найдены")
    print("[inputs] отчёт:", output)


if __name__ == "__main__":
    main()
