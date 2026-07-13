"""Стадия 06: честный blend с chunked BERT-предсказаниями."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("honest_bert_blend_079.py")
