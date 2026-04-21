r"""Register QuantPipe daily tasks in Windows Task Scheduler.

Creates two tasks under the "QuantPipe" folder:
  QuantPipe\DailyPipeline  — runs at 06:15 Mon-Fri (ingest + signals)
  QuantPipe\DailyRebalance — runs at 16:30 Mon-Fri (paper rebalance)

Run once as the user who will execute the tasks (no admin required for user tasks):
    python orchestration/setup_scheduler.py

To remove tasks:
    python orchestration/setup_scheduler.py --remove

To list registered QuantPipe tasks:
    schtasks /Query /TN "QuantPipe" /FO LIST
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PIPELINE_BAT = str(PROJECT_DIR / "orchestration" / "run_pipeline.bat")
REBALANCE_BAT = str(PROJECT_DIR / "orchestration" / "run_rebalance.bat")

TASKS = [
    {
        "name": r"QuantPipe\DailyPipeline",
        "script": PIPELINE_BAT,
        "time": "06:15",
        "description": "QuantPipe: ingest prices, compute features, generate signals",
    },
    {
        "name": r"QuantPipe\DailyRebalance",
        "script": REBALANCE_BAT,
        "time": "16:30",
        "description": "QuantPipe: paper rebalance and position reconciliation",
    },
]


def _run(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def create_tasks() -> None:
    for task in TASKS:
        print(f"Registering: {task['name']} at {task['time']} Mon-Fri")
        code, out = _run([
            "schtasks", "/Create",
            "/TN", task["name"],
            "/TR", task["script"],
            "/SC", "WEEKLY",
            "/D", "MON,TUE,WED,THU,FRI",
            "/ST", task["time"],
            "/F",    # overwrite if already exists
        ])
        if code == 0:
            print(f"  OK: {task['name']}")
        else:
            print(f"  FAILED (code={code}): {out}")


def remove_tasks() -> None:
    for task in TASKS:
        print(f"Removing: {task['name']}")
        code, out = _run(["schtasks", "/Delete", "/TN", task["name"], "/F"])
        if code == 0:
            print(f"  OK")
        else:
            print(f"  FAILED or not found: {out}")


def list_tasks() -> None:
    lines = []
    for task in TASKS:
        code, out = _run(["schtasks", "/Query", "/TN", task["name"], "/FO", "LIST"])
        lines.append(out if code == 0 else f"{task['name']}: not found")
    print("\n".join(lines) if lines else "No QuantPipe tasks found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage QuantPipe scheduled tasks")
    parser.add_argument("--remove", action="store_true", help="Remove all QuantPipe tasks")
    parser.add_argument("--list", action="store_true", help="List QuantPipe tasks")
    args = parser.parse_args()

    if args.remove:
        remove_tasks()
    elif args.list:
        list_tasks()
    else:
        create_tasks()
        print()
        list_tasks()
