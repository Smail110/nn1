"""РЎС‚Р°РґРёСЏ 08: fine-tuning raw-target RuBERT РЅР° РїРѕР»РЅРѕРј train."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_raw_rubert.py")

