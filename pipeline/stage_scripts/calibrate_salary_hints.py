"""РЎС‚Р°РґРёСЏ 07: supervised-РєР°Р»РёР±СЂРѕРІРєР° salary hints."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("calibrate_salary_hints.py")

