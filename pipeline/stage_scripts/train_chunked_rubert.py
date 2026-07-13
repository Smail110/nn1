"""РЎС‚Р°РґРёСЏ 05: РѕР±СѓС‡РµРЅРёРµ chunked RuBERT РЅР° РїРѕР»РЅРѕРј train."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_chunked_rubert.py")

