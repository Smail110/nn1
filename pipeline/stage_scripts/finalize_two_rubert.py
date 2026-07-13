"""РЎС‚Р°РґРёСЏ 12: blend РґРІСѓС… BERT РјРѕРґРµР»РµР№."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("finalize_two_rubert_blend.py")

