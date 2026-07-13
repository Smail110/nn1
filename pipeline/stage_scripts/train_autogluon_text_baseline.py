"""РЎС‚Р°РґРёСЏ 01: AutoGluon baseline РЅР° СЃС‹СЂРѕРј С‚РµРєСЃС‚Рµ Рё salary hints."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("train_autogluon_text_baseline.py")

