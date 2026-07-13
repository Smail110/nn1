"""РЎС‚Р°РґРёСЏ 23: С„РёРЅР°Р»СЊРЅС‹Р№ candidate reranker Рё submission.csv."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script("build_submission.py")

