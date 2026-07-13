"""Стадия 01: AutoGluon baseline на сыром тексте и salary hints."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("honest_raw_leak_autogluon.py")
