"""Стадия 19: local OOF для RuRoBERTa-large."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script(
    "train_rubert_large_081.py",
    "--stage",
    "local",
    "--tag",
    "roberta",
    "--base-model",
    "ruroberta_large_base",
)
