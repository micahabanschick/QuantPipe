#!/usr/bin/env python3
"""bootstrap.py — one-shot setup for a new QuantPipe installation.

Usage:
    python bootstrap.py                # dashboard-ready setup
    python bootstrap.py --live         # + IBKR live trading guidance
    python bootstrap.py --force-backfill   # re-download all price history
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    print("bootstrap.py requires Python 3.10+. Please upgrade Python and retry.")
    sys.exit(1)

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
BRONZE_EQUITY = DATA_DIR / "bronze" / "equity" / "daily"
BRONZE_CRYPTO = DATA_DIR / "bronze" / "crypto" / "daily"
GOLD_FEATURES = DATA_DIR / "gold" / "equity" / "features"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

_UV_WIN = Path.home() / ".local" / "bin" / "uv.exe"
_UV_UNIX = Path.home() / ".local" / "bin" / "uv"


# ── Output helpers ─────────────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    line = "=" * 62
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _ok(msg: str)   -> None: print(f"  [OK]   {msg}")
def _skip(msg: str) -> None: print(f"  [SKIP] {msg}")
def _info(msg: str) -> None: print(f"         {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL] {msg}", file=sys.stderr)


def _die(msg: str, code: int = 1) -> None:
    _fail(msg)
    sys.exit(code)


# ── uv helpers ─────────────────────────────────────────────────────────────────

def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (_UV_WIN, _UV_UNIX):
        if candidate.exists():
            return str(candidate)
    return None


def _install_uv() -> str:
    print("  uv not found — installing...")
    if platform.system() == "Windows":
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command",
             "irm https://astral.sh/uv/install.ps1 | iex"],
            capture_output=True, text=True,
        )
    else:
        result = subprocess.run(
            ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        _die(f"uv install failed:\n{result.stderr}")
    uv = _find_uv()
    if not uv:
        _die(
            "uv installed but executable not found in PATH.\n"
            "  Restart your shell and re-run: python bootstrap.py"
        )
    return uv


def _run(uv: str, *args: str) -> None:
    """Run a uv command, streaming output live. Exits on failure."""
    cmd = [uv, *args]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        _die(f"Command failed (code {result.returncode}): {' '.join(cmd)}")


def _run_python(uv: str, *script_args: str, check: bool = True) -> int:
    """Run a Python script via `uv run python`, streaming output live."""
    cmd = [uv, "run", "python", *script_args]
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode != 0:
        _die(f"Script failed: {' '.join(script_args)}")
    return result.returncode


def _has_parquet(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


# ── Setup steps ────────────────────────────────────────────────────────────────

def step_install(uv: str) -> None:
    _sep("Step 1/6 — Install dependencies")
    _info("uv sync --extra execution --extra portfolio --extra backtest")
    print()
    _run(uv, "sync", "--extra", "execution", "--extra", "portfolio", "--extra", "backtest")
    print()
    _ok("All dependencies installed")


def step_env() -> None:
    _sep("Step 2/6 — Environment file")
    if ENV_FILE.exists():
        _skip(".env already exists")
        return

    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        _ok(".env created from .env.example")
    else:
        _warn(".env.example not found — creating blank .env")
        ENV_FILE.touch()

    print()
    print("  Edit .env to configure credentials (all sections are optional")
    print("  until you need that feature):")
    print()
    print("    Dashboard + backtests (no keys required):")
    print("      yfinance and the Kraken public API work without credentials.")
    print()
    print("    Crypto live trading:")
    print("      KRAKEN_API_KEY + KRAKEN_SECRET")
    print()
    print("    IBKR paper / live trading:")
    print("      IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_PAPER")
    print("      (see --live flag for full IBKR setup instructions)")
    print()
    print("    Alerting (optional):")
    print("      PUSHOVER_TOKEN + PUSHOVER_USER  or  NTFY_TOPIC")


def step_backfill(uv: str, force: bool) -> None:
    _sep("Step 3/6 — Historical data backfill  (~5-15 min, one-time)")
    equity_ok = _has_parquet(BRONZE_EQUITY)
    crypto_ok = _has_parquet(BRONZE_CRYPTO)

    if equity_ok and crypto_ok and not force:
        _skip("Bronze data already present — skipping")
        _info("Pass --force-backfill to re-download everything")
        return

    print()
    if force or (not equity_ok and not crypto_ok):
        _run_python(uv, "orchestration/backfill_history.py")
    elif not equity_ok:
        _info("Crypto data found — backfilling equity only")
        _run_python(uv, "orchestration/backfill_history.py", "--asset-class", "equity")
    else:
        _info("Equity data found — backfilling crypto only")
        _run_python(uv, "orchestration/backfill_history.py", "--asset-class", "crypto")

    print()
    _ok("Backfill complete")


def step_features(uv: str, force: bool) -> None:
    _sep("Step 4/6 — Feature computation")
    if _has_parquet(GOLD_FEATURES) and not force:
        _skip("Gold features already present — skipping")
        _info("Pass --force-backfill to recompute")
        return
    print()
    _run_python(uv, "features/compute.py")
    print()
    _ok("Features written to gold layer")


def step_pipeline(uv: str) -> None:
    _sep("Step 5/6 — Generate initial signals")
    _info("Skipping live ingest — using backfilled data")
    print()
    rc = _run_python(uv, "orchestration/run_pipeline.py", "--skip-ingest", check=False)
    print()
    if rc == 0:
        _ok("Signals generated — target_weights.parquet and portfolio_log.parquet ready")
    elif rc == 1:
        _warn("Pipeline finished with partial failures — check logs/pipeline.log")
        _info("The app will still launch; some dashboard tiles may show warnings")
    else:
        _fail("Signal generation failed — check logs/signals.log")
        _info("The app will still launch; signal pages will show 'no data'")


def _activate_hint() -> str:
    """Return the correct venv activation command for the current shell."""
    if sys.platform == "win32":
        # Detect whether we're inside a POSIX shell (Git Bash, MSYS2, WSL)
        # or a native Windows shell (PowerShell / cmd).
        shell = os.environ.get("SHELL", "")          # set by bash/zsh, absent in PS/cmd
        comspec = os.environ.get("ComSpec", "")      # set by cmd/PowerShell, absent in bash
        in_posix_shell = bool(shell) or "bash" in os.environ.get("TERM_PROGRAM", "").lower()

        if in_posix_shell:
            return "source .venv/Scripts/activate   # Git Bash / MSYS2"
        else:
            return ".venv\\Scripts\\Activate.ps1    # PowerShell"
    else:
        return "source .venv/bin/activate"


def step_done(live: bool, launch: bool, uv: str) -> None:
    _sep("Step 6/6 — All done")
    _ok("QuantPipe is ready\n")

    activate_cmd = _activate_hint()
    print("  Activate the virtual environment (one-time per shell session):")
    print(f"    {activate_cmd}\n")
    print("  Then launch the dashboard:")
    print("    streamlit run app.py\n")
    print("  Or skip activation entirely — uv run always works:")
    print("    uv run streamlit run app.py\n")
    print("  Run the daily pipeline after markets close:")
    print("    uv run python orchestration/run_pipeline.py\n")
    print("  Automate (cron — 6 AM weekdays):")
    print("    0 6 * * 1-5  cd /path/to/QuantPipe && \\")
    print("                 .venv/Scripts/python.exe orchestration/run_pipeline.py\n")
    print("  Windows Task Scheduler equivalent:")
    print("    Program : .venv\\Scripts\\python.exe")
    print("    Args    : orchestration/run_pipeline.py")
    print("    Start in: C:\\path\\to\\QuantPipe")
    print("    Schedule: 6:15 AM, Monday-Friday\n")

    if launch:
        _sep("Launching dashboard")
        print()
        _info("Starting streamlit — press Ctrl+C to stop")
        print()
        subprocess.run([uv, "run", "streamlit", "run", "app.py"], cwd=ROOT)

    if live:
        _sep("IBKR Live Trading Setup")
        print()
        print("  1. Download and install TWS or IB Gateway:")
        print("     https://www.interactivebrokers.com/en/trading/tws.php\n")
        print("  2. In TWS/Gateway: File -> Global Configuration -> API -> Settings")
        print("       Enable ActiveX and Socket Clients : ON")
        print("       Port (pick one):")
        print("         7497  TWS paper    |  7496  TWS live")
        print("         4002  Gateway paper|  4001  Gateway live\n")
        print("  3. Set these values in .env:")
        print("       IBKR_HOST=127.0.0.1")
        print("       IBKR_PORT=4002        # match the port above")
        print("       IBKR_CLIENT_ID=1")
        print("       IBKR_PAPER=true       # change to false for real money\n")
        print("  4. Start TWS or Gateway, then execute a rebalance:")
        print("       uv run python orchestration/rebalance.py --broker ibkr\n")
        print("  IMPORTANT: always verify on paper (IBKR_PAPER=true) before")
        print("  switching to live. Paper and live use separate account numbers.\n")

    _sep()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Bootstrap a new QuantPipe installation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bootstrap.py                  # standard setup\n"
            "  python bootstrap.py --launch         # setup + open the app\n"
            "  python bootstrap.py --live           # + IBKR instructions\n"
            "  python bootstrap.py --force-backfill # re-download all history\n"
        ),
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Print IBKR live trading setup instructions at the end",
    )
    parser.add_argument(
        "--force-backfill", action="store_true",
        help="Re-download all price history and recompute features even if data exists",
    )
    parser.add_argument(
        "--launch", action="store_true",
        help="Launch the Streamlit dashboard immediately after setup completes",
    )
    args = parser.parse_args()

    _sep("QuantPipe Bootstrap")
    print()
    print("  This script will:")
    print("    1. Install uv (if needed) and sync all Python dependencies")
    print("    2. Create .env from .env.example (if not present)")
    print("    3. Backfill 7 years of price history  (equity ETFs + crypto)")
    print("    4. Compute features  (momentum, volatility, reversal, volume)")
    print("    5. Generate initial signals and portfolio weights")
    print("    6. Print next steps\n")
    print("  Safe to re-run — each step skips if its output already exists.")
    if args.force_backfill:
        print("  --force-backfill: will re-download and recompute everything.")
    if args.launch:
        print("  --launch: will start the Streamlit dashboard when setup completes.")
    print()

    uv = _find_uv()
    if uv is None:
        uv = _install_uv()
    else:
        _ok(f"uv found: {uv}")

    step_install(uv)
    step_env()
    step_backfill(uv, args.force_backfill)
    step_features(uv, args.force_backfill)
    step_pipeline(uv)
    step_done(args.live, args.launch, uv)


if __name__ == "__main__":
    main()
