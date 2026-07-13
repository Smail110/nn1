"""Стадия 04: KNN/text blend поверх salary leak baseline."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("knn_text_blend_079.py")
