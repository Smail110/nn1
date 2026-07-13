"""РЎС‚Р°РґРёСЏ 03: СЂРµРіСѓР»СЏСЂРЅС‹Рµ salary-hint РїСЂРёР·РЅР°РєРё РёР· С‚РµРєСЃС‚Р° РІР°РєР°РЅСЃРёР№."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("extract_salary_hints.py")

