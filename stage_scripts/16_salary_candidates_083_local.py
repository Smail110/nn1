"""Стадия 16: локальный salary-candidate classifier."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("broad_salary_candidates_083.py")
