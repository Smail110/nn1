"""Стадия 07: supervised-калибровка salary hints."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("supervised_salary_hint_081.py")
