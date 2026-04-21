"""Kill-switch support.

Place a file named QP_HALT in the project root to stop the pipeline
from doing any work at the next entry point check. Remove the file to
resume normal operation.

Usage in any orchestration entry point:
    from orchestration._halt import check_halt
    check_halt()   # raises SystemExit(3) if QP_HALT exists
"""

import logging
import sys
from pathlib import Path

from config.settings import PROJECT_ROOT

_HALT_FILE = PROJECT_ROOT / "QP_HALT"
log = logging.getLogger(__name__)


def check_halt() -> None:
    """Exit with code 3 if QP_HALT exists in the project root."""
    if _HALT_FILE.exists():
        log.critical(
            f"KILL-SWITCH ACTIVE — found {_HALT_FILE}. "
            "Remove the file to resume pipeline operations."
        )
        sys.exit(3)
