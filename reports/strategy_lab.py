"""Strategy Lab — code editor, backtester, parameter sweep, and walk-forward validation."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_ace import st_ace

from reports._theme import (
    COLORS,
    PLOTLY_CONFIG,
    apply_theme,
    kpi_card,
    page_header,
    section_label,
    status_banner,
)

_ROOT       = Path(__file__).parent.parent
_STRAT_DIR  = _ROOT / "strategies"
_FAILED_DIR = _STRAT_DIR / "_failed"
_ACE_THEME  = "tomorrow_night"

_ACE_LANG: dict[str, str] = {
    ".py":   "python",
    ".md":   "markdown",
    ".json": "json",
    ".toml": "toml",
    ".txt":  "text",
    ".yaml": "yaml",
    ".yml":  "yaml",
}

_MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Strategy templates ────────────────────────────────────────────────────────

_PY_TEMPLATE = """\
\"\"\"{name} — {description}

Required interface (tools/backtest_runner.py):
  NAME, DESCRIPTION, DEFAULT_PARAMS, get_signal(), get_weights()
\"\"\"

import polars as pl
from signals.momentum import cross_sectional_momentum, momentum_weights

NAME        = "{name}"
DESCRIPTION = "{description}"
DEFAULT_PARAMS = {{
    "lookback_years": 6,
    "top_n":          5,
    "cost_bps":       5.0,
    "weight_scheme":  "equal",
}}


def get_signal(features, rebal_dates, top_n=DEFAULT_PARAMS["top_n"], **kwargs):
    return cross_sectional_momentum(features, rebal_dates, top_n=top_n)


def get_weights(signal, weight_scheme=DEFAULT_PARAMS["weight_scheme"], **kwargs):
    return momentum_weights(signal, weight_scheme=weight_scheme)
"""

_README_TEMPLATE = """\
# {name}

> {description}

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical window for the backtest |
| `top_n` | 5 | Number of ETFs held at each rebalance |
| `cost_bps` | 5.0 | Round-trip transaction cost (basis points) |
| `weight_scheme` | equal | `equal` or `vol_scaled` |

## Signal

*(Describe the signal construction here.)*

## Known Limitations

*(List any known limitations here.)*
"""


# ── Folder helpers ────────────────────────────────────────────────────────────

def _migrate_legacy_strategies() -> None:
    _STRAT_DIR.mkdir(exist_ok=True)
    for py_file in list(_STRAT_DIR.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        slug = py_file.stem
        target_dir = _STRAT_DIR / slug
        if not target_dir.exists():
            target_dir.mkdir()
            py_file.rename(target_dir / py_file.name)


def _list_strategies() -> list[Path]:
    _STRAT_DIR.mkdir(exist_ok=True)
    _migrate_legacy_strategies()
    return [
        d for d in sorted(_STRAT_DIR.iterdir())
        if d.is_dir() and not d.name.startswith("_") and any(d.glob("*.py"))
    ]


def _main_py(strategy_dir: Path) -> Path | None:
    preferred = strategy_dir / f"{strategy_dir.name}.py"
    if preferred.exists():
        return preferred
    pys = sorted(strategy_dir.glob("*.py"))
    return pys[0] if pys else None


def _strategy_files(strategy_dir: Path) -> list[Path]:
    pys  = sorted(strategy_dir.glob("*.py"))
    rest = sorted(f for f in strategy_dir.iterdir() if f.is_file() and f.suffix != ".py")
    return pys + rest


def _display_name(strategy_dir: Path) -> str:
    main = _main_py(strategy_dir)
    if main:
        try:
            spec = importlib.util.spec_from_file_location("_tmp", main)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            name = getattr(mod, "NAME", None)
            if name:
                return name
        except Exception:
            pass
    return strategy_dir.name.replace("_", " ").title()


def _strategy_options(dirs: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in dirs:
        label = _display_name(d)
        if label in out:
            label = f"{label} ({d.name})"
        out[label] = d
    return out


def _name_to_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "strategy"


def _load_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"# File not found: {path}\n"


# ── Editor ────────────────────────────────────────────────────────────────────

def _render_editor(file_path: Path, strat_dir: Path | None = None) -> None:
    disk_key  = f"disk__{file_path}"
    edit_key  = f"edit__{file_path}"
    dirty_key = f"dirty_{file_path}"

    if disk_key not in st.session_state:
        content = _load_file(file_path)
        st.session_state[disk_key]  = content
        st.session_state[edit_key]  = content
        st.session_state[dirty_key] = False

    lang = _ACE_LANG.get(file_path.suffix, "text")

    # Include AI apply-version in the key so the widget re-mounts with new content
    # when the assistant's "Apply to Editor" button is pressed.
    apply_ver = st.session_state.get(
        f"ai_apply_ver_{strat_dir.name}" if strat_dir else "_noop", 0
    )

    new_content = st_ace(
        value=st.session_state[edit_key],
        language=lang,
        theme=_ACE_THEME,
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=True,
        min_lines=30,
        max_lines=55,
        key=f"ace_{file_path}_v{apply_ver}",
    )

    if new_content is not None:
        st.session_state[edit_key]  = new_content
        st.session_state[dirty_key] = (new_content != st.session_state[disk_key])

    dirty = st.session_state[dirty_key]

    ab1, ab2, ab3, ab4 = st.columns([2, 2, 2, 4])
    with ab1:
        save_clicked = st.button(
            "💾 Save", use_container_width=True, disabled=not dirty,
            type="primary" if dirty else "secondary", key=f"save_{file_path}",
        )
    with ab2:
        discard_clicked = st.button(
            "↩ Discard", use_container_width=True, disabled=not dirty,
            key=f"discard_{file_path}",
        )
    with ab3:
        reload_clicked = st.button(
            "🔄 Reload", use_container_width=True,
            help="Re-read from disk (discards unsaved edits)", key=f"reload_{file_path}",
        )
    with ab4:
        if dirty:
            st.markdown(
                f'<div style="padding:8px 0;color:{COLORS["warning"]};'
                f'font-size:0.80rem;font-weight:600;">● Unsaved changes</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="padding:8px 0;color:{COLORS["positive"]};font-size:0.80rem;">'
                f'✓ Saved</div>',
                unsafe_allow_html=True,
            )

    if save_clicked:
        try:
            file_path.write_text(st.session_state[edit_key], encoding="utf-8")
            st.session_state[disk_key]  = st.session_state[edit_key]
            st.session_state[dirty_key] = False
            st.success(f"Saved → strategies/{file_path.parent.name}/{file_path.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    if discard_clicked:
        st.session_state[edit_key]  = st.session_state[disk_key]
        st.session_state[dirty_key] = False
        st.rerun()

    if reload_clicked:
        fresh = _load_file(file_path)
        st.session_state[disk_key]  = fresh
        st.session_state[edit_key]  = fresh
        st.session_state[dirty_key] = False
        st.rerun()


# ── New-strategy dialog ───────────────────────────────────────────────────────

@st.dialog("New Strategy")
def _new_strategy_dialog() -> None:
    st.markdown(
        f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;margin-bottom:12px;">'
        "Creates a strategy folder with a Python file and README."
        "</div>", unsafe_allow_html=True,
    )
    name = st.text_input("Strategy name", placeholder="e.g. Momentum Vol-Scaled")
    desc = st.text_input("Description",   placeholder="One-line summary")
    slug = _name_to_slug(name) if name else "strategy"
    st.markdown(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.75rem;margin:4px 0 12px;">'
        f'Will be saved as <code>strategies/{slug}/</code></div>',
        unsafe_allow_html=True,
    )
    col_create, col_cancel = st.columns(2)
    with col_create:
        create = st.button("Create", type="primary", use_container_width=True, disabled=not name)
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    if create and name:
        target_dir = _STRAT_DIR / slug
        if target_dir.exists():
            st.error(f"strategies/{slug}/ already exists.")
        else:
            target_dir.mkdir(parents=True)
            description = desc or "No description provided."
            (target_dir / f"{slug}.py").write_text(
                _PY_TEMPLATE.format(name=name, description=description), encoding="utf-8",
            )
            (target_dir / "README.md").write_text(
                _README_TEMPLATE.format(name=name, description=description), encoding="utf-8",
            )
            st.session_state["selected_strategy"] = str(target_dir)
            st.rerun()


# ── Discard dialog ────────────────────────────────────────────────────────────

@st.dialog("Discard Strategy")
def _discard_strategy_dialog(strategy_dir: Path) -> None:
    display = _display_name(strategy_dir)
    st.markdown(
        f'<div style="color:{COLORS["warning"]};font-size:0.85rem;margin-bottom:10px;">'
        f'Move <strong>{display}</strong> to the failed strategies archive?'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.80rem;margin-bottom:16px;">'
        f'The strategy folder will be moved to <code>strategies/_failed/{strategy_dir.name}/</code>. '
        f'All files are preserved — you can restore it manually at any time.'
        f'</div>',
        unsafe_allow_html=True,
    )
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Move to Failed", type="primary", use_container_width=True):
            _FAILED_DIR.mkdir(parents=True, exist_ok=True)
            dest = _FAILED_DIR / strategy_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(strategy_dir), str(dest))
            st.session_state.pop("selected_strategy", None)
            st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ── AI coding assistant ───────────────────────────────────────────────────────

_AI_SYSTEM = """\
You are an expert quantitative finance developer embedded in QuantPipe's Strategy Lab.
Your job is to write, refactor, debug, and explain Python strategy code that runs inside QuantPipe.

## QuantPipe strategy interface

Every strategy is a plain Python module with these top-level attributes:

```python
NAME        : str          # display name shown in the UI
DESCRIPTION : str          # one-line summary
DEFAULT_PARAMS : dict      # must contain: lookback_years, top_n, cost_bps, weight_scheme

def get_signal(
    features    : pl.DataFrame,       # columns: rebalance_date | date, symbol,
                                      #   momentum_12m_1m, realized_vol_21d
    rebal_dates : list,               # monthly rebalance timestamps
    top_n       : int,
    prices_df   : pl.DataFrame | None,# columns: date, symbol, adj_close (or close)
    **kwargs,
) -> pl.DataFrame:
    # Must return columns: rebalance_date, symbol, score, rank, selected (bool), equity_pct (float 0-1)
    # When NO candidates pass filters → emit a __CASH__ row:
    #   {"rebalance_date": rd, "symbol": "__CASH__", "score": 0.0,
    #    "rank": 0, "selected": False, "equity_pct": 0.0}

def get_weights(
    signal       : pl.DataFrame,
    weight_scheme: str,
    **kwargs,
) -> pl.DataFrame:
    # Must return columns: rebalance_date, symbol, weight (float)
    # When selected is empty for a date → emit:
    #   {"rebalance_date": rd, "symbol": "__CASH__", "weight": 0.0}
    # equity_pct MUST be clipped: float(np.clip(equity_pct, 0.0, 1.0))
```

## Hard rules
- Long-only (no short selling, no negative weights)
- Monthly rebalancing only (rebal_dates is provided; do not generate your own dates)
- Use Polars DataFrames throughout; use pandas only for intermediate window math if needed
- The backtest engine forward-fills weights, so every rebalance date must appear in the
  weights output — even if the target is 100% cash (emit the __CASH__ sentinel)
- equity_pct < 1 means partial equity allocation; the engine preserves uninvested NAV as cash

## Style
- No docstrings beyond the module-level triple-quote block
- No inline comments unless the logic would surprise a reader
- Return pl.DataFrame() (empty) early if inputs are empty or missing required columns
"""


def _ai_api_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _render_ai_assistant(
    main_py: Path | None,
    strat_dir: Path,
    result_data: dict | None = None,
) -> None:
    """Full-width Claude chat panel for in-lab strategy coding help."""
    try:
        import anthropic as _anthropic
    except ImportError:
        st.warning(
            "Install the Anthropic SDK to enable the AI assistant: "
            "`pip install anthropic`"
        )
        return

    api_key = _ai_api_key()
    if not api_key:
        st.info(
            "Add your Anthropic API key to `.streamlit/secrets.toml` as "
            "`ANTHROPIC_API_KEY = \"sk-...\"` to enable the AI assistant."
        )
        return

    chat_key  = f"ai_chat_{strat_dir.name}"
    apply_ver = f"ai_apply_ver_{strat_dir.name}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    if apply_ver not in st.session_state:
        st.session_state[apply_ver] = 0

    messages: list[dict] = st.session_state[chat_key]

    # ── Current code injected as context ─────────────────────────────────────
    current_code = ""
    if main_py:
        edit_key = f"edit__{main_py}"
        current_code = st.session_state.get(edit_key) or _load_file(main_py)

    context_block = (
        f"\n\n## Current strategy file (`{main_py.name if main_py else 'none'}`)\n\n"
        f"```python\n{current_code}\n```"
        if current_code else ""
    )

    # ── Latest backtest result injected as context ────────────────────────────
    if result_data and result_data.get("ok"):
        m = result_data.get("metrics", {})
        def _fmt(v, pct=False):
            if v is None:
                return "—"
            return f"{float(v):+.2%}" if pct else f"{float(v):.3f}"
        context_block += (
            f"\n\n## Latest backtest result\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Sharpe | {_fmt(m.get('sharpe'))} |\n"
            f"| CAGR | {_fmt(m.get('cagr'), pct=True)} |\n"
            f"| Total Return | {_fmt(m.get('total_return'), pct=True)} |\n"
            f"| Max Drawdown | {_fmt(m.get('max_drawdown'), pct=True)} |\n"
            f"| Sortino | {_fmt(m.get('sortino'))} |\n"
            f"| Calmar | {_fmt(m.get('calmar'))} |\n"
            f"| Tracking Error | {_fmt(result_data.get('tracking_error'), pct=True)} |\n"
            f"| Information Ratio | {_fmt(result_data.get('information_ratio'))} |\n"
            f"| Trades | {m.get('n_trades', '—')} |\n"
        )

    # ── Chat history display ──────────────────────────────────────────────────
    history_container = st.container()
    with history_container:
        for idx, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                content = msg["content"]
                # Render markdown — code blocks come through naturally
                st.markdown(content)

                # "Apply to editor" button on assistant messages containing code
                if msg["role"] == "assistant" and main_py:
                    code_blocks = re.findall(
                        r"```python\n([\s\S]*?)```", content
                    )
                    if code_blocks:
                        if st.button(
                            "Apply to Editor",
                            key=f"ai_apply_{strat_dir.name}_{idx}",
                            help="Replace the editor content with the code above",
                        ):
                            edit_key  = f"edit__{main_py}"
                            dirty_key = f"dirty_{main_py}"
                            st.session_state[edit_key]  = code_blocks[-1]
                            st.session_state[dirty_key] = True
                            # Bump version so st_ace picks up the new value
                            st.session_state[apply_ver] += 1
                            st.rerun()

    # ── Clear button ─────────────────────────────────────────────────────────
    if messages:
        if st.button("Clear conversation", key=f"ai_clear_{strat_dir.name}"):
            st.session_state[chat_key] = []
            st.rerun()

    # ── Input ─────────────────────────────────────────────────────────────────
    prompt = st.chat_input(
        "Ask Claude to write, fix, or explain the strategy code...",
        key=f"ai_input_{strat_dir.name}",
    )
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build API messages — inject fresh code context into each user turn
    api_msgs: list[dict] = []
    for i, m in enumerate(messages):
        role    = m["role"]
        content = m["content"]
        # Append code context to the last user message only
        if role == "user" and i == len(messages) - 1:
            content = content + context_block
        api_msgs.append({"role": role, "content": content})

    client = _anthropic.Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        try:
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=_AI_SYSTEM,
                messages=api_msgs,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as exc:
            placeholder.error(f"API error: {exc}")
            return

        # Apply button for the freshly generated response
        code_blocks = re.findall(r"```python\n([\s\S]*?)```", full_response)
        if code_blocks and main_py:
            if st.button(
                "Apply to Editor",
                key=f"ai_apply_{strat_dir.name}_latest",
                help="Replace the editor content with the generated code",
                type="primary",
            ):
                edit_key  = f"edit__{main_py}"
                dirty_key = f"dirty_{main_py}"
                st.session_state[edit_key]  = code_blocks[-1]
                st.session_state[dirty_key] = True
                st.session_state[apply_ver] += 1
                st.rerun()

    messages.append({"role": "assistant", "content": full_response})
    st.session_state[chat_key] = messages


# ── Strategy validator ────────────────────────────────────────────────────────

def _validate_strategy(main_py: Path) -> list[str]:
    """Return list of issues, or [] if the strategy is clean."""
    issues = []
    try:
        source = main_py.read_text(encoding="utf-8")
        compile(source, str(main_py), "exec")
    except SyntaxError as e:
        issues.append(f"SyntaxError line {e.lineno}: {e.msg}")
        return issues

    try:
        spec = importlib.util.spec_from_file_location("_val", main_py)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        issues.append(f"Import error: {e}")
        return issues

    for attr in ("get_signal", "get_weights"):
        if not hasattr(mod, attr):
            issues.append(f"Missing required function: {attr}()")
    for attr in ("NAME", "DESCRIPTION", "DEFAULT_PARAMS"):
        if not hasattr(mod, attr):
            issues.append(f"Missing module attribute: {attr}")
    dp = getattr(mod, "DEFAULT_PARAMS", {})
    for key in ("top_n", "lookback_years", "cost_bps", "weight_scheme"):
        if key not in dp:
            issues.append(f"DEFAULT_PARAMS missing key: {key!r}")
    return issues


# ── Chart builders ────────────────────────────────────────────────────────────

def _equity_fig(data: dict) -> go.Figure:
    equity   = data.get("equity", {})
    bench    = data.get("benchmark", {})
    strat_nm = data.get("strategy_name", "Strategy")
    eq_vals  = equity.get("values", [])
    fig = go.Figure()
    if equity.get("dates"):
        eq_dates = pd.to_datetime(equity["dates"])
        fig.add_trace(go.Scatter(
            x=eq_dates, y=eq_vals, mode="lines", name=strat_nm,
            line=dict(color=COLORS["positive"], width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
            hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
        ))
        if bench.get("dates") and eq_vals:
            b_vals = bench["values"]
            scale  = eq_vals[0] / b_vals[0] if b_vals and b_vals[0] else 1
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(bench["dates"]), y=[v * scale for v in b_vals],
                mode="lines", name="SPY (scaled)",
                line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
                hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
            ))
    apply_theme(fig, title="Equity Curve vs. SPY", height=320)
    fig.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
    return fig


def _rolling_fig(data: dict) -> go.Figure:
    rs  = data.get("rolling_sharpe", {})
    win = rs.get("window", 252)
    fig = go.Figure()
    if rs.get("dates"):
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(rs["dates"]), y=rs["values"],
            name=f"Rolling Sharpe ({win}d)",
            line=dict(color=COLORS["info"], width=2),
            hovertemplate="%{x|%b %Y}<br>Sharpe %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=1.0, line=dict(color=COLORS["positive"], width=1, dash="dot"))
        fig.add_hline(y=0.0, line=dict(color=COLORS["neutral"],  width=1))
    apply_theme(fig, title=f"Rolling Sharpe ({win}-day window)", height=220)
    return fig


def _drawdown_fig(data: dict) -> go.Figure:
    dd = data.get("drawdown_pct", {})
    fig = go.Figure()
    if dd.get("dates"):
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dd["dates"]), y=dd["values"],
            fill="tozeroy", fillcolor="rgba(255,75,75,0.12)",
            line=dict(color=COLORS["negative"], width=1.5),
            name="Drawdown",
            hovertemplate="%{x|%b %d %Y}<br>%{y:.1f}%<extra></extra>",
        ))
    apply_theme(fig, title="Drawdown from Peak", height=180)
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


def _monthly_heatmap_fig(monthly_data: dict) -> go.Figure | None:
    if not monthly_data:
        return None
    years = sorted(monthly_data.keys())
    z, text = [], []
    for yr in years:
        row_z, row_t = [], []
        for m in _MONTH_ORDER:
            v = monthly_data[yr].get(m)
            row_z.append(v if v is not None else float("nan"))
            row_t.append(f"{v*100:+.1f}%" if v is not None else "")
        z.append(row_z)
        text.append(row_t)
    flat = [v for row in z for v in row if v == v]
    max_abs = max((abs(v) for v in flat), default=0.05)
    fig = go.Figure(go.Heatmap(
        z=z, x=_MONTH_ORDER, y=years,
        text=text, texttemplate="%{text}", textfont=dict(size=10, color=COLORS["text"]),
        colorscale=[[0.0,"#ff4b4b"],[0.5,COLORS["card_bg"]],[1.0,"#00d4aa"]],
        zmin=-max_abs, zmax=max_abs, zmid=0,
        showscale=True,
        colorbar=dict(tickformat=".0%", len=0.8, thickness=12,
                      tickfont=dict(color=COLORS["neutral"], size=10)),
        hovertemplate="<b>%{y} %{x}</b><br>%{text}<extra></extra>",
    ))
    apply_theme(fig, title="Monthly Returns", height=max(200, len(years) * 32 + 80))
    fig.update_layout(
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=50, r=60, t=60, b=10),
    )
    return fig


# ── Results renderer ──────────────────────────────────────────────────────────

def _show_result_tabs(data: dict, result_key: str) -> None:
    if not data or not data.get("ok"):
        return

    metrics  = data.get("metrics", {})
    params   = data.get("params", {})
    strat_nm = data.get("strategy_name", "Strategy")
    sharpe   = float(metrics.get("sharpe") or 0)

    if sharpe >= 0.8:
        bc, bt = COLORS["positive"], f"Backtest complete — Sharpe {sharpe:.2f}"
    elif sharpe >= 0.4:
        bc, bt = COLORS["warning"],  f"Backtest complete — Sharpe {sharpe:.2f} (below target)"
    else:
        bc, bt = COLORS["negative"], f"Backtest complete — Sharpe {sharpe:.2f} (low)"
    st.markdown(status_banner(bt, bc), unsafe_allow_html=True)

    p_period = f"{metrics.get('start','?')} → {metrics.get('end','?')} ({metrics.get('years','?')}y)"
    p_cfg    = (f"Top-{params.get('top_n','?')} · {params.get('weight_scheme','?')} · "
                f"{params.get('cost_bps','?')} bps")
    st.markdown(
        f'<div style="color:{COLORS["neutral"]};font-size:0.75rem;margin-bottom:10px;">'
        f'<b style="color:{COLORS["text"]}">{strat_nm}</b> &nbsp;·&nbsp; '
        f'{p_period} &nbsp;·&nbsp; {p_cfg}</div>',
        unsafe_allow_html=True,
    )

    def _pct(v): return f"{v:+.1%}" if isinstance(v, float) else "—"
    def _f(v):   return f"{v:.3f}"  if isinstance(v, float) else "—"

    # 8 KPI cards in two rows of 4
    kpis = [
        ("Total Return",  _pct(metrics.get("total_return")), None),
        ("CAGR",          _pct(metrics.get("cagr")), None),
        ("Sharpe",        _f(sharpe), COLORS["positive"] if sharpe >= 0.8 else COLORS["warning"]),
        ("Sortino",       _f(metrics.get("sortino")), None),
        ("Max Drawdown",  _pct(metrics.get("max_drawdown")), COLORS["negative"]),
        ("Calmar",        _f(metrics.get("calmar")), None),
        ("Alpha (ann)",   _pct(data.get("alpha", 0.0)), None),
        ("Beta vs SPY",   f"{data.get('beta', 1.0):.2f}", None),
    ]
    for i in range(0, len(kpis), 4):
        cols = st.columns(4)
        for col, (label, val, accent) in zip(cols, kpis[i:i+4]):
            with col:
                st.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.72rem;margin:4px 0 8px;">'
        f'Tracking error: {data.get("tracking_error", 0):.2%} &nbsp;·&nbsp; '
        f'Information ratio: {data.get("information_ratio", 0):.2f} &nbsp;·&nbsp; '
        f'Trades: {metrics.get("n_trades","—")} &nbsp;·&nbsp; '
        f'Total cost: ${metrics.get("total_cost", 0):,.0f}</div>',
        unsafe_allow_html=True,
    )

    # CSV exports
    eq = data.get("equity", {})
    tlog = data.get("trade_log", [])
    _dl_cols = st.columns([2, 2, 6])
    if eq.get("dates"):
        csv = pd.DataFrame({"date": eq["dates"], "nav": eq["values"]}).to_csv(index=False).encode()
        _dl_cols[0].download_button(
            "⬇ Equity Curve (CSV)", csv,
            file_name=f"{strat_nm.replace(' ','_')}_equity.csv",
            mime="text/csv", key=f"dl_{result_key}",
        )
    if tlog:
        tdf_csv = pd.DataFrame(tlog).to_csv(index=False).encode()
        _dl_cols[1].download_button(
            "⬇ Trade Log (CSV)", tdf_csv,
            file_name=f"{strat_nm.replace(' ','_')}_trades.csv",
            mime="text/csv", key=f"dl_trades_{result_key}",
        )

    st.divider()

    t1, t2, t3, t4 = st.tabs(["Equity Curve", "Rolling Analytics", "Monthly Returns", "Trade Log"])

    with t1:
        st.plotly_chart(_equity_fig(data), config=PLOTLY_CONFIG, use_container_width=True)

    with t2:
        st.plotly_chart(_rolling_fig(data),   config=PLOTLY_CONFIG, use_container_width=True)
        st.plotly_chart(_drawdown_fig(data),  config=PLOTLY_CONFIG, use_container_width=True)

    with t3:
        fig_hm = _monthly_heatmap_fig(data.get("monthly_returns", {}))
        if fig_hm:
            st.plotly_chart(fig_hm, config=PLOTLY_CONFIG, use_container_width=True)
        else:
            st.info("Need at least 2 months of data for the monthly heatmap.")

    with t4:
        tlog = data.get("trade_log", [])
        if tlog:
            tdf = pd.DataFrame(tlog)
            b  = int((tdf["side"] == "BUY").sum())  if "side"  in tdf.columns else 0
            s  = int((tdf["side"] == "SELL").sum()) if "side"  in tdf.columns else 0
            av = float(abs(tdf["value"]).mean())     if "value" in tdf.columns else 0.0
            c1, c2, c3 = st.columns(3)
            c1.metric("Buy orders",      b)
            c2.metric("Sell orders",     s)
            c3.metric("Avg trade value", f"${av:,.0f}")
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded.")


# ── Parameter sweep ───────────────────────────────────────────────────────────

def _run_sweep(
    main_py: Path,
    top_n_list: list[int],
    lookback_list: list[int],
    cost_bps: float,
    weight_scheme: str,
) -> pd.DataFrame:
    """Grid search over top_n × lookback_years. Returns Sharpe DataFrame."""
    import polars as pl
    from backtest.engine import run_backtest
    from features.compute import load_features
    from signals.momentum import get_monthly_rebalance_dates
    from storage.parquet_store import load_bars
    from storage.universe import universe_as_of_date

    spec = importlib.util.spec_from_file_location("_sweep", main_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    end        = date.today()
    max_lb     = max(lookback_list)
    start_max  = date(end.year - max_lb, end.month, end.day)
    symbols    = universe_as_of_date("equity", end, require_data=True)
    prices_all = load_bars(symbols, start_max, end, "equity")
    feats_all  = load_features(symbols, start_max, end, "equity",
                                feature_list=["momentum_12m_1m", "realized_vol_21d"])
    all_dates  = sorted(prices_all["date"].unique().to_list())

    results: dict[tuple, float] = {}
    for lb in lookback_list:
        start_lb = date(end.year - lb, end.month, end.day)
        p_lb = prices_all.filter(pl.col("date") >= start_lb)
        f_lb = feats_all.filter(pl.col("date") >= start_lb)
        rdates = get_monthly_rebalance_dates(start_lb, end, all_dates)
        for tn in top_n_list:
            try:
                sig = mod.get_signal(f_lb, rdates, top_n=tn, prices_df=p_lb)
                wts = mod.get_weights(sig, weight_scheme=weight_scheme)
                res = run_backtest(p_lb, wts, cost_bps=cost_bps)
                results[(lb, tn)] = res.sharpe
            except Exception:
                results[(lb, tn)] = float("nan")

    grid = pd.DataFrame(index=lookback_list, columns=top_n_list, dtype=float)
    for (lb, tn), sh in results.items():
        grid.loc[lb, tn] = sh
    grid.index.name   = "Lookback (years)"
    grid.columns.name = "Top-N"
    return grid


def _render_sweep_tab(main_py: Path, default_params: dict) -> None:
    st.markdown(
        f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;margin-bottom:12px;">'
        "Grid search over Lookback × Top-N to find the parameter set with the highest Sharpe. "
        "Each cell is a full backtest — keep the grid small (≤20 cells)."
        "</div>", unsafe_allow_html=True,
    )
    sa, sb = st.columns(2)
    with sa:
        top_n_str = st.text_input("Top-N values (comma-separated)", value="3,5,7,10", key="sw_tn")
    with sb:
        lb_str = st.text_input("Lookback years (comma-separated)", value="3,5,7", key="sw_lb")

    sc, sd = st.columns(2)
    with sc:
        sw_cost = st.number_input("Cost (bps)", value=float(default_params.get("cost_bps", 5.0)),
                                   min_value=0.0, max_value=100.0, step=0.5, key="sw_cost")
    with sd:
        sw_scheme_opts = ["equal", "vol_scaled"]
        sw_scheme = st.selectbox("Weighting", options=sw_scheme_opts,
                                  index=sw_scheme_opts.index(default_params.get("weight_scheme","equal")),
                                  key="sw_scheme")

    run_sweep = st.button("▶ Run Parameter Sweep", type="primary", key="run_sweep_btn")
    sweep_key = f"sweep_{main_py}"

    if run_sweep and main_py:
        try:
            tn_list = [int(x.strip()) for x in top_n_str.split(",") if x.strip()]
            lb_list = [int(x.strip()) for x in lb_str.split(",")    if x.strip()]
        except ValueError:
            st.error("Invalid grid values — use comma-separated integers.")
            return
        n_cells = len(tn_list) * len(lb_list)
        with st.spinner(f"Grid search ({n_cells} backtests)…"):
            try:
                grid = _run_sweep(main_py, tn_list, lb_list, sw_cost, sw_scheme)
                st.session_state[sweep_key] = grid
            except Exception as exc:
                st.error(f"Sweep failed: {exc}")
                return

    grid = st.session_state.get(sweep_key)
    if grid is not None and isinstance(grid, pd.DataFrame):
        best_idx    = grid.stack().idxmax()
        best_lb, best_tn = best_idx
        best_sh = grid.loc[best_lb, best_tn]
        st.success(f"Best: lookback={best_lb}y, top_n={best_tn} → Sharpe {best_sh:.3f}")

        z     = grid.values.tolist()
        x_lab = [str(c) for c in grid.columns.tolist()]
        y_lab = [str(i) for i in grid.index.tolist()]
        text  = [[f"{v:.2f}" if v == v else "" for v in row] for row in z]
        vmax  = float(np.nanmax(grid.values))
        vmin  = float(np.nanmin(grid.values))

        fig = go.Figure(go.Heatmap(
            z=z, x=x_lab, y=y_lab,
            text=text, texttemplate="%{text}", textfont=dict(size=11),
            colorscale=[[0.0,"#ff4b4b"],[0.5,COLORS["card_bg"]],[1.0,"#00d4aa"]],
            zmid=(vmax + vmin) / 2,
            colorbar=dict(title="Sharpe", thickness=12, len=0.8,
                          tickfont=dict(color=COLORS["neutral"], size=10)),
            hovertemplate="Lookback=%{y}y  Top-N=%{x}<br>Sharpe=%{text}<extra></extra>",
        ))
        apply_theme(fig, title="Sharpe Heatmap — Lookback × Top-N", height=300)
        fig.update_layout(
            xaxis=dict(title="Top-N positions"),
            yaxis=dict(title="Lookback years", autorange="reversed"),
        )
        st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)
        st.dataframe(grid.style.format("{:.3f}"), use_container_width=True)
    elif not run_sweep:
        st.info("Configure the grid and click **Run Parameter Sweep**.")


# ── Walk-forward validation ────────────────────────────────────────────────────

def _run_walk_forward(
    main_py: Path,
    top_n: int,
    lookback_years: int,
    cost_bps: float,
    weight_scheme: str,
    test_frac: float = 0.30,
) -> dict:
    """Split history into IS/OOS at test_frac; return metrics + equity curves."""
    import polars as pl
    from backtest.engine import run_backtest
    from features.compute import load_features
    from signals.momentum import get_monthly_rebalance_dates
    from storage.parquet_store import load_bars
    from storage.universe import universe_as_of_date

    spec = importlib.util.spec_from_file_location("_wf", main_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    end        = date.today()
    start_full = date(end.year - lookback_years, end.month, end.day)
    symbols    = universe_as_of_date("equity", end, require_data=True)
    prices     = load_bars(symbols, start_full, end, "equity")
    features   = load_features(symbols, start_full, end, "equity",
                                feature_list=["momentum_12m_1m", "realized_vol_21d"])
    all_dates  = sorted(prices["date"].unique().to_list())

    split_i  = int(len(all_dates) * (1 - test_frac))
    split_dt = all_dates[split_i]

    def _run_period(p_start, p_end):
        px = prices.filter((pl.col("date") >= p_start) & (pl.col("date") <= p_end))
        ft = features.filter((pl.col("date") >= p_start) & (pl.col("date") <= p_end))
        pd_ = sorted(px["date"].unique().to_list())
        rd  = get_monthly_rebalance_dates(p_start, p_end, pd_)
        sig = mod.get_signal(ft, rd, top_n=top_n, prices_df=px)
        wts = mod.get_weights(sig, weight_scheme=weight_scheme)
        return run_backtest(px, wts, cost_bps=cost_bps)

    res_is  = _run_period(start_full, split_dt)
    res_oos = _run_period(split_dt, end)

    def _eq(res):
        eq = res.equity_curve
        return {"dates": [str(d.date()) for d in eq.index],
                "values": [round(v, 2) for v in eq.tolist()]}

    return {
        "split_date": str(split_dt),
        "is_metrics":  {"sharpe": res_is.sharpe,  "cagr": res_is.cagr,
                         "max_drawdown": res_is.max_drawdown,  "total_return": res_is.total_return},
        "oos_metrics": {"sharpe": res_oos.sharpe, "cagr": res_oos.cagr,
                         "max_drawdown": res_oos.max_drawdown, "total_return": res_oos.total_return},
        "is_equity":  _eq(res_is),
        "oos_equity": _eq(res_oos),
    }


def _render_wf_tab(main_py: Path, default_params: dict) -> None:
    st.markdown(
        f'<div style="color:{COLORS["neutral"]};font-size:0.82rem;margin-bottom:12px;">'
        "Split the backtest period into in-sample (IS) and out-of-sample (OOS). "
        "A robust strategy should not severely degrade on OOS data."
        "</div>", unsafe_allow_html=True,
    )
    wa, wb, wc = st.columns(3)
    with wa:
        wf_lb   = st.number_input("Lookback (years)", min_value=2, max_value=20,
                                   value=int(default_params.get("lookback_years", 6)), key="wf_lb")
    with wb:
        wf_tn   = st.number_input("Top-N", min_value=1, max_value=20,
                                   value=int(default_params.get("top_n", 5)), key="wf_tn")
    with wc:
        wf_frac = st.slider("OOS fraction", min_value=0.10, max_value=0.50,
                             value=0.30, step=0.05, key="wf_frac")

    wf_cost = st.number_input("Cost (bps)", min_value=0.0, max_value=100.0,
                               value=float(default_params.get("cost_bps", 5.0)),
                               step=0.5, key="wf_cost")
    run_wf  = st.button("▶ Run Walk-Forward", type="primary", key="run_wf_btn")
    wf_key  = f"wf_{main_py}"

    if run_wf and main_py:
        with st.spinner("Running in-sample and out-of-sample backtests…"):
            try:
                wf_data = _run_walk_forward(
                    main_py, top_n=wf_tn, lookback_years=wf_lb,
                    cost_bps=wf_cost,
                    weight_scheme=default_params.get("weight_scheme", "equal"),
                    test_frac=wf_frac,
                )
                st.session_state[wf_key] = wf_data
            except Exception as exc:
                st.error(f"Walk-forward failed: {exc}")
                return

    wf = st.session_state.get(wf_key)
    if wf is None:
        st.info("Configure above and click **Run Walk-Forward**.")
        return

    split_dt = wf.get("split_date", "?")
    is_m     = wf.get("is_metrics",  {})
    oos_m    = wf.get("oos_metrics", {})

    st.markdown(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;'
        f'margin-bottom:8px;">Split date: {split_dt}</div>',
        unsafe_allow_html=True,
    )

    def _d(v): return f"{v:+.1%}" if isinstance(v, float) else "—"
    def _f(v): return f"{v:.3f}"  if isinstance(v, float) else "—"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IS Sharpe",       _f(is_m.get("sharpe")),
              delta=f"OOS {_f(oos_m.get('sharpe'))}", delta_color="normal")
    m2.metric("IS CAGR",         _d(is_m.get("cagr")),
              delta=f"OOS {_d(oos_m.get('cagr'))}", delta_color="normal")
    m3.metric("IS Max DD",       _d(is_m.get("max_drawdown")),
              delta=f"OOS {_d(oos_m.get('max_drawdown'))}", delta_color="inverse")
    m4.metric("IS Total Return", _d(is_m.get("total_return")),
              delta=f"OOS {_d(oos_m.get('total_return'))}", delta_color="normal")

    # Indexed equity curves overlaid
    fig = go.Figure()
    for label, color, eq_data in [
        ("In-Sample",     COLORS["positive"], wf.get("is_equity",  {})),
        ("Out-of-Sample", COLORS["warning"],  wf.get("oos_equity", {})),
    ]:
        if eq_data.get("dates"):
            vals = eq_data["values"]
            norm = [v / vals[0] * 100 for v in vals]
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(eq_data["dates"]), y=norm,
                name=label, line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>%{{x|%b %Y}}<br>%{{y:.1f}}<extra></extra>",
            ))

    fig.add_vline(
        x=pd.Timestamp(split_dt).timestamp() * 1000,
        line=dict(color=COLORS["info"], width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=pd.Timestamp(split_dt), y=105,
        text="IS | OOS", showarrow=False,
        font=dict(color=COLORS["info"], size=11),
        bgcolor=COLORS["card_bg"], bordercolor=COLORS["info"],
        borderwidth=1, borderpad=3,
    )
    apply_theme(fig, title="Walk-Forward: IS vs OOS (indexed to 100)", height=320)
    st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(page_header(
    "Strategy Lab",
    "Write, backtest, and optimise signal strategies with integrated parameter sweep and walk-forward validation.",
), unsafe_allow_html=True)

# ── Strategy selector ─────────────────────────────────────────────────────────

st.markdown(section_label("Strategy"), unsafe_allow_html=True)

strategy_dirs    = _list_strategies()
strategy_options = _strategy_options(strategy_dirs)

sel_col, new_col, disc_col = st.columns([5, 1, 1])

with new_col:
    if st.button("➕ New", use_container_width=True, help="Create a new strategy from template"):
        _new_strategy_dialog()

if not strategy_options:
    st.info("No strategies found in `strategies/`. Click **➕ New** to create your first one.")
    st.stop()

_saved_path = st.session_state.get("selected_strategy")
labels      = list(strategy_options.keys())
default_idx = 0
if _saved_path:
    for i, d in enumerate(strategy_options.values()):
        if str(d) == _saved_path:
            default_idx = i
            break

with sel_col:
    selected_label = st.selectbox(
        "Strategy", options=labels, index=default_idx, label_visibility="collapsed",
    )

selected_dir = strategy_options[selected_label]
st.session_state["selected_strategy"] = str(selected_dir)
main_py = _main_py(selected_dir)

with disc_col:
    if st.button(
        "🗑 Discard", use_container_width=True,
        help="Move this strategy to strategies/_failed/ (preserved, not deleted)",
    ):
        _discard_strategy_dialog(selected_dir)

st.markdown(
    f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
    f'border-left:4px solid {COLORS["positive"]};border-radius:0 8px 8px 0;'
    f'padding:10px 18px;margin:4px 0 18px;display:flex;align-items:baseline;gap:14px;">'
    f'<span style="font-size:1.05rem;font-weight:700;color:{COLORS["text"]};'
    f'letter-spacing:-0.02em;">{selected_label}</span>'
    f'<span style="font-size:0.72rem;color:{COLORS["text_muted"]};">'
    f'strategies/{selected_dir.name}/</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Two-column: editor | config + controls ────────────────────────────────────

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(section_label("Code Editor"), unsafe_allow_html=True)
    strat_files = _strategy_files(selected_dir)
    if not strat_files:
        st.warning("No files found in this strategy folder.")
    elif len(strat_files) == 1:
        _render_editor(strat_files[0], strat_dir=selected_dir)
    else:
        file_tabs = st.tabs([f.name for f in strat_files])
        for tab, file_path in zip(file_tabs, strat_files):
            with tab:
                _render_editor(file_path, strat_dir=selected_dir)

with right:
    st.markdown(section_label("Backtest Configuration"), unsafe_allow_html=True)

    _defaults: dict = {}
    if main_py:
        try:
            _spec = importlib.util.spec_from_file_location("_strat_tmp", main_py)
            _mod  = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _defaults = getattr(_mod, "DEFAULT_PARAMS", {})
        except Exception:
            pass

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        lookback_years = st.number_input(
            "Lookback (years)", min_value=1, max_value=20,
            value=int(_defaults.get("lookback_years", 6)), step=1,
        )
        cost_bps = st.number_input(
            "Cost (bps)", min_value=0.0, max_value=100.0,
            value=float(_defaults.get("cost_bps", 5.0)), step=0.5, format="%.1f",
        )
    with cfg2:
        top_n = st.number_input(
            "Top-N positions", min_value=1, max_value=20,
            value=int(_defaults.get("top_n", 5)), step=1,
        )
        scheme_opts   = ["equal", "vol_scaled"]
        scheme_def    = _defaults.get("weight_scheme", "equal")
        weight_scheme = st.selectbox(
            "Weighting", options=scheme_opts,
            index=scheme_opts.index(scheme_def) if scheme_def in scheme_opts else 0,
        )

    btn_run, btn_val = st.columns(2)
    with btn_run:
        run_btn = st.button(
            "▶ Run Backtest", type="primary", use_container_width=True, disabled=main_py is None,
        )
    with btn_val:
        val_btn = st.button(
            "✔ Validate", use_container_width=True, disabled=main_py is None,
            help="Check syntax, imports, and required interface",
        )

    if val_btn and main_py:
        issues = _validate_strategy(main_py)
        if issues:
            for iss in issues:
                st.error(iss)
        else:
            st.success("Strategy is valid — syntax, imports, and interface all pass.")

    st.divider()

    results_ph = st.empty()
    console_ph = st.empty()
    result_key = f"lab_result_{selected_dir.name}"

    # Show previous error if not re-running
    if result_key in st.session_state and not run_btn:
        prev = st.session_state[result_key]
        if prev and not prev.get("ok"):
            with results_ph.container():
                err = prev.get("error", "Unknown error")
                st.markdown(
                    status_banner(f"Backtest failed — {err[:160]}", COLORS["negative"]),
                    unsafe_allow_html=True,
                )
            if prev.get("_console"):
                with console_ph.container():
                    with st.expander("Console output", expanded=True):
                        st.code("\n".join(prev["_console"]), language="text")

    if run_btn and main_py:
        runner = str(_ROOT / "tools" / "backtest_runner.py")
        cmd = [
            sys.executable, runner,
            "--strategy",       str(main_py),
            "--lookback-years", str(int(lookback_years)),
            "--top-n",          str(int(top_n)),
            "--cost-bps",       str(float(cost_bps)),
            "--weight-scheme",  weight_scheme,
        ]
        with results_ph.container():
            st.markdown(status_banner("Backtest running…", COLORS["info"], animate=True),
                        unsafe_allow_html=True)

        stdout_buf, stderr_lines = "", []
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", cwd=str(_ROOT),
            )
            stdout_buf, stderr_raw = proc.communicate(timeout=300)
            stderr_lines = [ln for ln in stderr_raw.splitlines() if ln.strip()]
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_buf, _ = proc.communicate()
            stderr_lines.append("ERROR: Backtest timed out after 5 minutes.")
        except Exception as exc:
            stderr_lines.append(f"ERROR launching subprocess: {exc}")

        result_data = {"ok": False, "error": "No JSON output from runner"}
        for line in stdout_buf.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    result_data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass

        result_data["_console"] = stderr_lines
        st.session_state[result_key] = result_data

        if result_data.get("ok"):
            history_key = f"lab_history_{selected_dir.name}"
            hist = st.session_state.get(history_key, [])
            _m = result_data.get("metrics", {})
            hist.append({
                "Run #":    len(hist) + 1,
                "Date":     date.today().isoformat(),
                "Sharpe":   round(float(_m.get("sharpe") or 0), 3),
                "CAGR":     f"{float(_m.get('cagr') or 0):+.1%}",
                "Max DD":   f"{float(_m.get('max_drawdown') or 0):+.1%}",
                "Top-N":    int(top_n),
                "Lookback": int(lookback_years),
                "Cost bps": float(cost_bps),
            })
            st.session_state[history_key] = hist[-5:]

        if not result_data.get("ok"):
            with results_ph.container():
                err = result_data.get("error", "Unknown error")
                st.markdown(status_banner(f"Backtest failed — {err[:160]}", COLORS["negative"]),
                            unsafe_allow_html=True)
            if stderr_lines:
                with console_ph.container():
                    with st.expander("Console output", expanded=True):
                        st.code("\n".join(stderr_lines), language="text")
        else:
            results_ph.empty()
            console_ph.empty()
            if stderr_lines:
                with console_ph.container():
                    with st.expander("Console output", expanded=False):
                        st.code("\n".join(stderr_lines), language="text")

# ── Full-width result tabs ────────────────────────────────────────────────────

result_data = st.session_state.get(result_key, {})
if result_data and result_data.get("ok"):
    st.divider()
    _show_result_tabs(result_data, result_key)

    # ── Promote to Portfolio button ───────────────────────────────────────────
    _pm_col, _ = st.columns([3, 9])
    with _pm_col:
        if st.button(
            "🚀 Promote to Portfolio",
            type="primary",
            use_container_width=True,
            help="Write active_strategy.json — run the pipeline to apply",
        ):
            try:
                _cfg_dir = _ROOT / "config"
                _cfg_dir.mkdir(exist_ok=True)
                _promo = {
                    "strategy_name": selected_label,
                    "strategy_path": str(main_py),
                    "params": {
                        "top_n": int(top_n),
                        "lookback_years": int(lookback_years),
                        "cost_bps": float(cost_bps),
                        "weight_scheme": weight_scheme,
                    },
                    "promoted_at": date.today().isoformat(),
                    "sharpe": float(result_data.get("metrics", {}).get("sharpe") or 0),
                }
                (_cfg_dir / "active_strategy.json").write_text(
                    json.dumps(_promo, indent=2), encoding="utf-8"
                )
                st.success(
                    f"Promoted **{selected_label}** → `config/active_strategy.json`. "
                    "Run `uv run python orchestration/generate_signals.py` to apply."
                )
            except Exception as _pe:
                st.error(f"Promote failed: {_pe}")

# ── Backtest History ──────────────────────────────────────────────────────────

_hist_key = f"lab_history_{selected_dir.name}"
_hist     = st.session_state.get(_hist_key, [])
if len(_hist) >= 2:
    st.divider()
    st.markdown(section_label(f"Backtest History — {selected_label}"), unsafe_allow_html=True)
    _hist_df = pd.DataFrame(_hist)
    st.dataframe(_hist_df, hide_index=True, width="stretch")

# ── AI Coding Assistant ───────────────────────────────────────────────────────

st.divider()
st.markdown(section_label("AI Coding Assistant"), unsafe_allow_html=True)
st.markdown(
    f'<div style="color:{COLORS["text_muted"]};font-size:0.80rem;margin:-6px 0 14px;">'
    f'Claude has full context of the current strategy file. Ask it to write, '
    f'fix, or explain code — then click <strong>Apply to Editor</strong> to '
    f'insert the result directly into the editor above.'
    f'</div>',
    unsafe_allow_html=True,
)
_render_ai_assistant(main_py, selected_dir, result_data=result_data)

# ── Advanced analysis ─────────────────────────────────────────────────────────

st.divider()
st.markdown(section_label("Advanced Analysis"), unsafe_allow_html=True)

adv1, adv2 = st.tabs(["Parameter Sweep", "Walk-Forward Validation"])

with adv1:
    if main_py:
        _render_sweep_tab(main_py, _defaults)
    else:
        st.info("Select a strategy to use the parameter sweep.")

with adv2:
    if main_py:
        _render_wf_tab(main_py, _defaults)
    else:
        st.info("Select a strategy to use walk-forward validation.")
