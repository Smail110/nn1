"""Стадия 11: обучение второй BERT модели на полном train."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_full_second_bert_081.py")
