"""РЎС‚Р°РґРёСЏ 15: Р»РѕРєР°Р»СЊРЅС‹Рµ TF-IDF salary-level РїСЂРёР·РЅР°РєРё."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_tfidf_salary_levels.py")

