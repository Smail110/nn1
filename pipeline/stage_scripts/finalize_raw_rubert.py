"""РЎС‚Р°РґРёСЏ 09: СЃР±РѕСЂРєР° raw BERT submission 081."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("finalize_raw_rubert.py")

