"""РЎС‚Р°РґРёСЏ 21: full-train TF-IDF salary-level РјРѕРґРµР»СЊ."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_full_tfidf_levels.py")

