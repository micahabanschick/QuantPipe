"""Master pipeline orchestrator — chains all daily steps.

Chain: ingest_daily -> generate_signals -> alert on any failure.

Each step exits non-zero on failure. This script collects exit codes,
sends a single alert summarising any failures, and exits non-zero if
any step failed so the cron daemon records the failure.

Cron example (runs at 06:00 every weekday):
    0 6 * * 1-5 cd /path/to/QuantPipe && .venv/Scripts/python.exe orchestration/run_pipeline.py >> logs/pipeline.log 2>&1

Usage:
    uv run python orchestration/run_pipeline.py
    uv run python orchestration/run_pipeline.py --skip-ingest   # signals-only rerun
"""

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import LOGS_DIR, PROJECT_ROOT
from orchestration._halt import check_halt
from orchestration.generate_signals import run_generate_signals

_HEARTBEAT_PATH = PROJECT_ROOT / ".pipeline_heartbeat.json"


def _write_heartbeat(status: str, failures: list[str], elapsed: float, as_of: date) -> None:
    """Write a machine-readable heartbeat so dashboards don't need to grep logs."""
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": str(as_of),
        "status": status,           # "ok" | "partial" | "failed"
        "failures": failures,
        "elapsed_s": round(elapsed, 1),
    }
    try:
        tmp = _HEARTBEAT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        import os; os.replace(tmp, _HEARTBEAT_PATH)
    except Exception as exc:
        log.warning(f"Failed to write pipeline heartbeat: {exc}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


def _send_alert(message: str) -> None:
    from config.settings import NTFY_TOPIC, PUSHOVER_TOKEN, PUSHOVER_USER
    if PUSHOVER_TOKEN and PUSHOVER_USER:
        try:
            import requests
            requests.post("https://api.pushover.net/1/messages.json", data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message,
                "title": "QuantPipe Pipeline",
            }, timeout=10)
        except Exception as exc:
            log.warning(f"Pushover alert failed: {exc}")
    elif NTFY_TOPIC:
        try:
            import requests
            requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=message.encode(), timeout=10)
        except Exception as exc:
            log.warning(f"ntfy alert failed: {exc}")
    log.info(f"ALERT sent: {message}")


def _run_step(name: str, fn, *args, **kwargs) -> tuple[int, float]:
    """Run one pipeline step, return (exit_code, elapsed_seconds)."""
    log.info(f"--- Step: {name} ---")
    t0 = time.monotonic()
    try:
        code = fn(*args, **kwargs)
        elapsed = time.monotonic() - t0
        status = "OK" if code == 0 else f"FAILED (code={code})"
        log.info(f"--- {name}: {status} in {elapsed:.1f}s ---")
        return code, elapsed
    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.error(f"--- {name}: EXCEPTION in {elapsed:.1f}s: {exc} ---")
        return 2, elapsed


def run_pipeline(skip_ingest: bool = False, as_of: date | None = None) -> int:
    check_halt()   # abort immediately if QP_HALT file exists

    today = as_of or date.today()
    t_start = time.monotonic()
    failures: list[str] = []

    log.info(f"======== QuantPipe pipeline | {today} ========")

    # Step 1 — ingest
    if not skip_ingest:
        from orchestration.ingest_daily import main as ingest_main
        code, elapsed = _run_step("ingest_daily", ingest_main)
        if code != 0:
            failures.append(f"ingest_daily (code={code})")
            if code == 2:
                _send_alert(f"[{today}] Pipeline aborted: ingest_daily total failure.")
                log.error("Aborting pipeline — ingest totally failed.")
                return 2
    else:
        log.info("--- Step: ingest_daily SKIPPED ---")

    # Step 2 — generate signals
    code, elapsed = _run_step("generate_signals", run_generate_signals, today)
    if code != 0:
        failures.append(f"generate_signals (code={code})")

    # Step 3 — pull scheduled FRED series (best-effort, never aborts pipeline)
    from config.settings import FRED_API_KEY
    if FRED_API_KEY:
        from orchestration.pull_fred import main as pull_fred_main
        code, _ = _run_step("pull_fred", pull_fred_main)
        if code != 0:
            log.warning("FRED pull had errors — check logs/fred.log (pipeline continues)")
    else:
        log.info("--- Step: pull_fred SKIPPED (no FRED_API_KEY) ---")

    # Step 4 — pull macro regime indicators (best-effort, never aborts pipeline)
    if FRED_API_KEY:
        from orchestration.pull_macro import main as pull_macro_main
        code, _ = _run_step("pull_macro", pull_macro_main)
        if code != 0:
            log.warning("Macro pull had errors — check logs (pipeline continues)")
    else:
        log.info("--- Step: pull_macro SKIPPED (no FRED_API_KEY) ---")

    # Step 5 — pull earnings surprise data (best-effort, never aborts pipeline)
    from config.settings import ALPHA_VANTAGE_API_KEY
    if ALPHA_VANTAGE_API_KEY:
        from orchestration.pull_earnings import main as pull_earnings_main
        code, _ = _run_step("pull_earnings", pull_earnings_main)
        if code != 0:
            log.warning("Earnings pull had errors — check logs (pipeline continues)")
    else:
        log.info("--- Step: pull_earnings SKIPPED (no ALPHA_VANTAGE_API_KEY) ---")

    # Summary
    total_elapsed = time.monotonic() - t_start
    log.info(f"======== Pipeline complete in {total_elapsed:.1f}s | "
             f"{len(failures)} failure(s) ========")

    if failures:
        summary = ", ".join(failures)
        _write_heartbeat("partial" if len(failures) < 2 else "failed", failures, total_elapsed, today)
        _send_alert(f"[{today}] Pipeline finished with failures: {summary}. Check logs.")
        return 1

    _write_heartbeat("ok", [], total_elapsed, today)
    _send_alert_success(today, total_elapsed)
    return 0


def _send_alert_success(today: date, elapsed: float) -> None:
    """Send a success notification (only if alerting is configured)."""
    from config.settings import NTFY_TOPIC, PUSHOVER_TOKEN, PUSHOVER_USER
    if not (PUSHOVER_TOKEN or NTFY_TOPIC):
        return
    _send_alert(f"[{today}] Pipeline OK in {elapsed:.0f}s. Signals updated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full QuantPipe daily pipeline")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion and run signals only")
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="Override today's date (for backfill/testing)")
    args = parser.parse_args()
    sys.exit(run_pipeline(skip_ingest=args.skip_ingest, as_of=args.date))


if __name__ == "__main__":
    main()
