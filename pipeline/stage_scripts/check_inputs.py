"""РЎС‚Р°РґРёСЏ 00: РїСЂРѕРІРµСЂРєР° РІС…РѕРґРЅС‹С… С„Р°Р№Р»РѕРІ Рё РѕС‚РєСЂС‹С‚С‹С… pretrained-РјРѕРґРµР»РµР№.

Р­С‚Р° СЃС‚Р°РґРёСЏ РЅРµ РѕР±СѓС‡Р°РµС‚ РјРѕРґРµР»СЊ. РћРЅР° С„РёРєСЃРёСЂСѓРµС‚, С‡С‚Рѕ pipeline СЃС‚Р°СЂС‚СѓРµС‚ С‚РѕР»СЊРєРѕ РѕС‚
РёСЃС…РѕРґРЅС‹С… train/test/sample Рё РѕС‚ Р»РѕРєР°Р»СЊРЅРѕ СЃРѕС…СЂР°РЅС‘РЅРЅС‹С… РѕС‚РєСЂС‹С‚С‹С… СЏР·С‹РєРѕРІС‹С… РјРѕРґРµР»РµР№.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import ARTIFACT_DIR, DEEPPAVLOV_RUBERT_BASE, OPEN_PRETRAINED_MODEL_DIRS, RAW_DATA_FILES, ROOT, SOURCE_ROOT


# РќСѓР¶РµРЅ РґР»СЏ С‡РёС‚Р°РµРјС‹С… СЂСѓСЃСЃРєРёС… СЃРѕРѕР±С‰РµРЅРёР№ РІ Windows PowerShell.
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
        print("[inputs] РќРµ С…РІР°С‚Р°РµС‚ РѕР±СЏР·Р°С‚РµР»СЊРЅС‹С… РІС…РѕРґРѕРІ:")
        for item in missing:
            print("  -", item)
        raise SystemExit(2)

    print("[inputs] train/test/sample Рё open pretrained РјРѕРґРµР»Рё РЅР°Р№РґРµРЅС‹")
    print("[inputs] РѕС‚С‡С‘С‚:", output)


if __name__ == "__main__":
    main()

