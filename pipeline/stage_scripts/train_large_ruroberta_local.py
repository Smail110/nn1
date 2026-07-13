"""РЎС‚Р°РґРёСЏ 19: local OOF РґР»СЏ RuRoBERTa-large."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline_utils import run_parent_script


run_parent_script(
    "train_large_transformer.py",
    "--stage",
    "local",
    "--tag",
    "roberta",
    "--base-model",
    "ruroberta_large_base",
)

