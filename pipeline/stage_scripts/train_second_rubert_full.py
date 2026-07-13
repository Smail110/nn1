"""РЎС‚Р°РґРёСЏ 11: РѕР±СѓС‡РµРЅРёРµ РІС‚РѕСЂРѕР№ BERT РјРѕРґРµР»Рё РЅР° РїРѕР»РЅРѕРј train."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_second_rubert.py")

