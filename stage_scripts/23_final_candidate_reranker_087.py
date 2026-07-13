"""Стадия 23: финальный candidate reranker и submission.csv."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("stage_scripts/build_final_submission.py")
