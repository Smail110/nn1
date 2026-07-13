"""РЎС‚Р°РґРёСЏ 06: С‡РµСЃС‚РЅС‹Р№ blend СЃ chunked BERT-РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏРјРё."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("blend_chunked_rubert.py")

