"""РЎС‚Р°РґРёСЏ 16: Р»РѕРєР°Р»СЊРЅС‹Р№ salary-candidate classifier."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_salary_candidate_classifier.py")

