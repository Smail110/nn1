"""Стадия 14: финальный boost 081 с salary-hint classifier."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("finalize_hint_classifier_081.py")
