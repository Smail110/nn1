"""РЎС‚Р°РґРёСЏ 04: KNN/text blend РїРѕРІРµСЂС… salary leak baseline."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("blend_knn_text.py")

