"""РЎС‚Р°РґРёСЏ 22: full-train salary-candidate classifier."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_full_salary_candidates.py")

