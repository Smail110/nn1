"""Стадия 02: TF-IDF + target encoding baseline/stack."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("experiment_079.py")
