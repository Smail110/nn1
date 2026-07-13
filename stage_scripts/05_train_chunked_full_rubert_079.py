"""Стадия 05: обучение chunked RuBERT на полном train."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("1535.py")
