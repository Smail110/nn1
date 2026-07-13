"""Главный запуск полного salary prediction pipeline.

Пример полного запуска:

    python leaderboard_087_full_pipeline/run_all.py --from-scratch --stream

В режиме `--from-scratch` выбранные стадии запускаются заново.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from config import LOG_DIR, ROOT
from pipeline_utils import ensure_dirs, missing, read_tail, rel, validate_final_submission
from stages import STAGES, Stage


# Чтобы русские сообщения нормально отображались в логах и терминале Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def stage_index(name: str) -> int:
    """Находит стадию по полному имени или префиксу."""

    for index, stage in enumerate(STAGES):
        if stage.name == name or stage.name.startswith(name):
            return index
    raise SystemExit(f"Неизвестная стадия: {name}")


def run_stage(stage: Stage, *, rerun: bool, stream: bool, dry_run: bool) -> None:
    """Запускает одну стадию и проверяет её outputs."""

    if stage.is_done() and not rerun:
        print(f"[SKIP] {stage.name}: outputs already exist", flush=True)
        return

    missing_inputs = missing(stage.input_paths())
    if missing_inputs:
        label = "[WOULD-BLOCK]" if dry_run else "[BLOCKED]"
        print(f"{label} {stage.name}: не хватает входов")
        for path in missing_inputs:
            print("  -", rel(path))
        if dry_run:
            print("  dry-run продолжается: в реальном полном прогоне эти файлы создадут предыдущие стадии.")
        else:
            raise SystemExit(2)

    print(f"[RUN] {stage.name}", flush=True)
    print("     ", " ".join(stage.command), flush=True)
    if stage.note:
        print("     ", stage.note, flush=True)
    if dry_run:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{stage.name}.stdout.log"
    stderr_path = LOG_DIR / f"{stage.name}.stderr.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    started = time.time()
    if stream:
        completed = subprocess.run(stage.command, cwd=ROOT, env=env, text=True)
        stdout_path.write_text("Вывод был показан напрямую в консоли.\n", encoding="utf-8")
        stderr_path.write_text("Вывод был показан напрямую в консоли.\n", encoding="utf-8")
    else:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(stage.command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, text=True)

    elapsed = (time.time() - started) / 60
    print(f"[DONE] {stage.name}: exit={completed.returncode}, elapsed={elapsed:.1f} min", flush=True)

    if completed.returncode != 0:
        print(f"\n--- stdout tail: {stdout_path} ---")
        print(read_tail(stdout_path))
        print(f"\n--- stderr tail: {stderr_path} ---")
        print(read_tail(stderr_path))
        raise SystemExit(completed.returncode)

    missing_outputs = missing(stage.output_paths())
    if missing_outputs:
        print(f"[BLOCKED] {stage.name}: стадия завершилась, но outputs не найдены")
        for path in missing_outputs:
            print("  -", rel(path))
        raise SystemExit(3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Полный запуск pipeline для прогноза зарплаты.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="Показать стадии и выйти")
    parser.add_argument("--resume", action="store_true", help="Пропускать готовые стадии")
    parser.add_argument("--rerun", action="store_true", help="Перезапускать стадии даже при готовых outputs")
    parser.add_argument("--from-scratch", action="store_true", help="Пересчитать выбранные стадии с включёнными тяжёлыми этапами")
    parser.add_argument("--include-heavy", action="store_true", help="Разрешить тяжёлые AutoGluon/BERT/Transformer стадии")
    parser.add_argument("--stream", action="store_true", help="Показывать output стадий в консоли")
    parser.add_argument("--dry-run", action="store_true", help="Показать план без запуска")
    parser.add_argument("--start-at", default=None, help="Начать со стадии по имени/префиксу")
    parser.add_argument("--stop-after", default=None, help="Остановиться после стадии по имени/префиксу")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.from_scratch:
        args.rerun = True
        args.include_heavy = True
    if args.rerun and args.resume:
        raise SystemExit("Нельзя одновременно --resume и --rerun")
    if not args.rerun:
        args.resume = True

    if args.list:
        for number, stage in enumerate(STAGES, start=1):
            status = "done" if stage.is_done() else "missing"
            heavy = ", heavy" if stage.heavy else ""
            print(f"{number:02d}. {stage.name} [{status}{heavy}]")
            print("    ", stage.note)
            if stage.outputs:
                print("     outputs:", ", ".join(stage.outputs))
        return

    start = stage_index(args.start_at) if args.start_at else 0
    stop = stage_index(args.stop_after) if args.stop_after else len(STAGES) - 1
    selected = STAGES[start : stop + 1]

    print("Full salary prediction pipeline", flush=True)
    print("Root:", ROOT, flush=True)
    print("Mode:", "from-scratch" if args.from_scratch else ("rerun" if args.rerun else "resume"), flush=True)
    print(flush=True)

    for stage in selected:
        if stage.heavy and not args.include_heavy and not stage.is_done() and not args.dry_run:
            print(f"[HEAVY-BLOCKED] {stage.name}")
            print("  Это тяжёлая стадия, а outputs отсутствуют.")
            print("  Запусти с --include-heavy или --from-scratch, если точно хочешь обучать.")
            raise SystemExit(4)
        run_stage(stage, rerun=args.rerun, stream=args.stream, dry_run=args.dry_run)

    if not args.dry_run and stop == len(STAGES) - 1:
        validate_final_submission()


if __name__ == "__main__":
    main()
