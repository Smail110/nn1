"""РЎС‚Р°РґРёСЏ 13: РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ РЅР°РґС‘Р¶РЅРѕСЃС‚Рё salary hints."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_salary_hint_classifier.py")

