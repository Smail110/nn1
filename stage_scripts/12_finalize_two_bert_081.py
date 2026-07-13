"""Стадия 12: blend двух BERT моделей."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("finalize_two_bert_081.py")
