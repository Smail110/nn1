"""Стадия 03: регулярные salary-hint признаки из текста вакансий."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("salary_leak_v2.py")
