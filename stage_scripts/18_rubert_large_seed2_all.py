"""Стадия 18: fine-tuning RuBERT-large, второй seed."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script(
    "train_rubert_large_081.py",
    "--stage",
    "all",
    "--tag",
    "seed2",
    "--seed",
    "991",
    "--base-model",
    "rubert_large_base",
)
