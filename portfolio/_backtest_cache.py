"""Backtest result cache — keyed by strategy slug.

Cache dir: data/gold/equity/backtest_cache/<slug>.json
Invalidated when:
  - strategy .py file is newer than the cache
  - cache is older than 24 hours
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from config.settings import DATA_DIR

CACHE_DIR = DATA_DIR / "gold" / "equity" / "backtest_cache"
_TTL = 86_400  # 24 hours


@dataclass
class CachedResult:
    slug: str
    name: str
    equity_dates: list
    equity_values: list
    benchmark_dates: list
    benchmark_values: list
    metrics: dict
    params: dict
    cached_at: str  # ISO 8601


def _path(slug: str) -> Path:
    return CACHE_DIR / f"{slug}.json"


def is_valid(slug: str, strategy_path: Path) -> bool:
    p = _path(slug)
    if not p.exists():
        return False
    cache_mtime = p.stat().st_mtime
    if time.time() - cache_mtime > _TTL:
        return False
    try:
        if strategy_path.stat().st_mtime > cache_mtime:
            return False
    except FileNotFoundError:
        return False
    return True


def load(slug: str) -> Optional[CachedResult]:
    p = _path(slug)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return CachedResult(**data)
    except Exception:
        return None


def save(result: CachedResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(result.slug)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    os.replace(tmp, p)


def invalidate(slug: str) -> None:
    p = _path(slug)
    if p.exists():
        p.unlink()
