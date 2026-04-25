"""Pull scheduled FRED series and save to data/alt/.

Called automatically at the end of run_pipeline.py if FRED_API_KEY is set
and data/alt/.fred_schedule.json contains series IDs.

Usage:
    uv run python orchestration/pull_fred.py
"""
import json
import logging
import sys
from datetime import date
from pathlib import Path

from config.settings import DATA_DIR, FRED_API_KEY, LOGS_DIR
from data_adapters.fred_adapter import FREDAdapter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "fred.log"),
    ],
)
log = logging.getLogger(__name__)

ALT_DIR    = DATA_DIR / "alt"
SCHED_FILE = ALT_DIR / ".fred_schedule.json"
META_FILE  = ALT_DIR / ".meta.json"


def _load(path: Path) -> dict | list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [] if "schedule" in path.name else {}


def main() -> int:
    if not FRED_API_KEY:
        log.info("FRED_API_KEY not set — skipping FRED pull")
        return 0

    sched = _load(SCHED_FILE)
    if not sched:
        log.info("No FRED series scheduled — skipping")
        return 0

    ALT_DIR.mkdir(parents=True, exist_ok=True)
    fred = FREDAdapter(FRED_API_KEY)
    meta = _load(META_FILE)
    errors = 0

    log.info(f"Pulling {len(sched)} scheduled FRED series…")
    for sid in sched:
        out = ALT_DIR / f"fred_{sid.lower()}.parquet"
        # Determine start: extend existing data or default to 2015
        existing_end = meta.get(f"fred_{sid.lower()}", {}).get("date_to", "2015-01-01")
        try:
            df = fred.get_series(sid, start=existing_end, end=str(date.today()))
            if df.is_empty():
                log.warning(f"  {sid}: no data returned")
                continue
            if out.exists():
                import polars as pl
                old = pl.read_parquet(out)
                df  = pl.concat([old, df]).unique(subset=["date"], keep="last").sort("date")
            df.write_parquet(out)
            meta[f"fred_{sid.lower()}"] = {
                "source": "FRED", "series_id": sid,
                "last_refreshed": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                "rows": len(df),
                "date_from": str(df["date"].min()),
                "date_to":   str(df["date"].max()),
            }
            log.info(f"  {sid}: {len(df)} rows saved")
        except Exception as exc:
            log.error(f"  {sid}: FAILED — {exc}")
            errors += 1

    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"FRED pull complete ({len(sched) - errors}/{len(sched)} succeeded)")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
