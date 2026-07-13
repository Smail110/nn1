"""Стадия 08: fine-tuning raw-target RuBERT на полном train."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_full_raw_bert_081.py")
