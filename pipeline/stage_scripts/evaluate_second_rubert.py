"""РЎС‚Р°РґРёСЏ 10: local-РѕС†РµРЅРєР° РІС‚РѕСЂРѕР№ BERT РјРѕРґРµР»Рё."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("evaluate_second_rubert.py")

