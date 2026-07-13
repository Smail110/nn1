"""Стадия 15: локальные TF-IDF salary-level признаки."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("tfidf_salary_levels_083.py")
