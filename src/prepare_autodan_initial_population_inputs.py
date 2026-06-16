"""Prepare defense inputs from the AutoDAN initial DAN-like population.

This script reads the official AutoDAN ``assets/prompt_group.pth`` file without
requiring PyTorch. The file is a PyTorch zip archive whose data payload is a
plain pickle list of strings. It combines the first N population members with
the first N AdvBench goals and writes this project's normalized JSONL input.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTODAN_ROOT = PROJECT_ROOT / "references" / "AutoDAN"
DEFAULT_POPULATION = AUTODAN_ROOT / "assets" / "prompt_group.pth"
DEFAULT_ADV_BENCH = AUTODAN_ROOT / "data" / "advbench" / "harmful_behaviors.csv"
DEFAULT_GCG_INPUT = PROJECT_ROOT / "results" / "archive" / "normalized_inputs" / "gcg_qwen25_15b.jsonl"
DEFAULT_ARCHIVE_OUTPUT = PROJECT_ROOT / "results" / "archive" / "normalized_inputs" / "autodan_initial_population_25.jsonl"
DEFAULT_TRAINING_OUTPUT = PROJECT_ROOT / "results" / "defense_training_inputs.jsonl"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "results" / "defense_training_inputs_summary.json"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "defense_training_inputs_summary.csv"


def read_autodan_population(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        data_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        population = pickle.load(io.BytesIO(archive.read(data_name)))
    if not isinstance(population, list) or not all(isinstance(item, str) for item in population):
        raise TypeError(f"Unexpected AutoDAN population format in {path}")
    return population


def read_advbench_goals(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected: list[dict[str, str]] = []
    for index, row in enumerate(rows[:limit]):
        selected.append(
            {
                "index": str(index),
                "goal": str(row.get("goal") or "").strip(),
                "target": str(row.get("target") or "").strip(),
            }
        )
    return selected


def build_full_prompt(suffix: str, goal: str) -> str:
    if "[REPLACE]" in suffix:
        return suffix.replace("[REPLACE]", goal.lower())
    return f"{suffix}\n\nUser request: {goal}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_gcg_training_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path)):
        rows.append(
            {
                "id": f"gcg_{row.get('index', index)}",
                "row_id": index,
                "split": "train",
                "task": "jailbreak_attack_detection",
                "label": "attack",
                "is_attack": True,
                "attack_type": "gcg",
                "attack_success": row.get("success"),
                "behavior_id": row.get("behavior_id"),
                "goal": row.get("goal"),
                "input": row.get("prompt"),
                "response": row.get("response"),
                "source": row.get("source_artifact") or str(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--advbench", type=Path, default=DEFAULT_ADV_BENCH)
    parser.add_argument("--gcg-input", type=Path, default=DEFAULT_GCG_INPUT)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--archive-output", type=Path, default=DEFAULT_ARCHIVE_OUTPUT)
    parser.add_argument("--training-output", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args()

    population = read_autodan_population(args.population)
    if len(population) < args.limit:
        raise ValueError(f"Population only has {len(population)} rows, cannot select {args.limit}.")
    goals = read_advbench_goals(args.advbench, args.limit)
    if len(goals) < args.limit:
        raise ValueError(f"AdvBench only has {len(goals)} rows, cannot select {args.limit}.")
    if not args.gcg_input.exists():
        raise FileNotFoundError(f"GCG input not found: {args.gcg_input}")

    archived_rows: list[dict[str, Any]] = []
    autodan_training_rows: list[dict[str, Any]] = []
    for index, (suffix, goal_row) in enumerate(zip(population[: args.limit], goals)):
        goal = goal_row["goal"]
        full_prompt = build_full_prompt(suffix, goal)
        archived_rows.append(
            {
                "index": index,
                "behavior_id": f"advbench_{index}",
                "goal": goal,
                "target": goal_row["target"],
                "final_suffix": suffix,
                "full_prompt": full_prompt,
                "final_response": None,
                "success": None,
                "attack_type": "autodan_initial_population",
                "source_artifact": str(args.population),
                "source_dataset": str(args.advbench),
            }
        )
        autodan_training_rows.append(
            {
                "id": f"autodan_initial_{index}",
                "row_id": index,
                "split": "train",
                "task": "jailbreak_attack_detection",
                "label": "attack",
                "is_attack": True,
                "attack_type": "autodan_initial_population",
                "attack_success": None,
                "behavior_id": f"advbench_{index}",
                "goal": goal,
                "input": full_prompt,
                "response": None,
                "source": str(args.population),
            }
        )

    write_jsonl(args.archive_output, archived_rows)

    gcg_training_rows = build_gcg_training_rows(args.gcg_input)
    training_rows = gcg_training_rows + autodan_training_rows
    for row_id, row in enumerate(training_rows):
        row["row_id"] = row_id
    write_jsonl(args.training_output, training_rows)

    summary = {
        "gcg_source": str(args.gcg_input),
        "population_source": str(args.population),
        "advbench_source": str(args.advbench),
        "limit": args.limit,
        "population_rows_available": len(population),
        "rows_written": len(training_rows),
        "gcg_rows": len(gcg_training_rows),
        "autodan_rows": len(autodan_training_rows),
        "outputs": {
            "archive": str(args.archive_output),
            "training": str(args.training_output),
        },
        "checks": {
            "unique_suffixes": len({row["final_suffix"] for row in archived_rows}),
            "rows_with_replace_placeholder": sum("[REPLACE]" in row["final_suffix"] for row in archived_rows),
            "empty_prompts": sum(not row["input"] for row in training_rows),
        },
    }
    summary_rows = [
        {
            "attack_type": "gcg",
            "rows": len(gcg_training_rows),
            "success_true": sum(row.get("attack_success") is True for row in gcg_training_rows),
            "success_false": sum(row.get("attack_success") is False for row in gcg_training_rows),
            "success_missing": sum(row.get("attack_success") is None for row in gcg_training_rows),
        },
        {
            "attack_type": "autodan_initial_population",
            "rows": len(autodan_training_rows),
            "success_true": 0,
            "success_false": 0,
            "success_missing": len(autodan_training_rows),
        }
    ]
    write_csv(args.summary_csv, summary_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
