"""Strategy Lab — code editor, strategy selector, and backtest runner."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

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

# ── Templates ─────────────────────────────────────────────────────────────────

_PY_TEMPLATE = '''\
"""{name} — {description}

Strategy interface (required by tools/backtest_runner.py):
  NAME          : displayed in the Strategy Lab selector
  DESCRIPTION   : one-line summary
  DEFAULT_PARAMS: fallback values used when the UI does not override them
  get_signal()  : (features, rebal_dates, **params) -> signal DataFrame
  get_weights() : (signal, **params) -> weights DataFrame
"""

import polars as pl

from signals.momentum import cross_sectional_momentum, momentum_weights

NAME = "{name}"
DESCRIPTION = "{description}"
DEFAULT_PARAMS = {{
    "lookback_years": 6,
    "top_n": 5,
    "cost_bps": 5.0,
    "weight_scheme": "equal",
}}


def get_signal(
    features: pl.DataFrame,
    rebal_dates: list,
    top_n: int = DEFAULT_PARAMS["top_n"],
    **kwargs,
) -> pl.DataFrame:
    """Rank symbols by 12-1 momentum and select the top_n on each rebalance date."""
    return cross_sectional_momentum(features, rebal_dates, top_n=top_n)


def get_weights(
    signal: pl.DataFrame,
    weight_scheme: str = DEFAULT_PARAMS["weight_scheme"],
    **kwargs,
) -> pl.DataFrame:
    """Convert ranked signal to target weights."""
    return momentum_weights(signal, weight_scheme=weight_scheme)
'''

_README_TEMPLATE = '''\
# {name}

> {description}

## Strategy Overview

*(Describe the strategy logic here.)*

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 6 | Historical period used for the backtest |
| `top_n` | 5 | Number of top-ranked ETFs to hold |
| `cost_bps` | 5.0 | Round-trip transaction cost in basis points |
| `weight_scheme` | equal | Position sizing — `equal` or `vol_scaled` |

## Signal

*(Describe the signal construction here.)*

## Known Limitations

*(List any known limitations here.)*

## References

*(Add references here.)*
'''


# ── Folder-based strategy helpers ─────────────────────────────────────────────

def _migrate_legacy_strategies() -> None:
    """Wrap any top-level flat .py files (old format) into their own subfolder."""
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
    """Return strategy directories that contain at least one .py file."""
    _STRAT_DIR.mkdir(exist_ok=True)
    _migrate_legacy_strategies()
    dirs = []
    for d in sorted(_STRAT_DIR.iterdir()):
        if d.is_dir() and any(d.glob("*.py")):
            dirs.append(d)
    return dirs


def _main_py(strategy_dir: Path) -> Path | None:
    """The primary .py file — same name as the folder, or the first .py found."""
    preferred = strategy_dir / f"{strategy_dir.name}.py"
    if preferred.exists():
        return preferred
    pys = sorted(strategy_dir.glob("*.py"))
    return pys[0] if pys else None


def _strategy_files(strategy_dir: Path) -> list[Path]:
    """All editable files: .py files first, then others alphabetically."""
    pys   = sorted(strategy_dir.glob("*.py"))
    rest  = sorted(f for f in strategy_dir.iterdir() if f.is_file() and f.suffix != ".py")
    return pys + rest


def _display_name(strategy_dir: Path) -> str:
    """Read NAME from the main .py; fall back to title-cased folder name."""
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
    """Map display label → directory Path."""
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


# ── Single-file editor ────────────────────────────────────────────────────────

def _render_editor(file_path: Path) -> None:
    """Render Ace editor + save/discard/reload bar for a single file."""
    disk_key  = f"disk__{file_path}"
    edit_key  = f"edit__{file_path}"
    dirty_key = f"dirty_{file_path}"

    if disk_key not in st.session_state:
        content = _load_file(file_path)
        st.session_state[disk_key]  = content
        st.session_state[edit_key]  = content
        st.session_state[dirty_key] = False

    lang = _ACE_LANG.get(file_path.suffix, "text")

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
        min_lines=28,
        max_lines=52,
        key=f"ace_{file_path}",
    )

    if new_content is not None:
        st.session_state[edit_key]  = new_content
        st.session_state[dirty_key] = (new_content != st.session_state[disk_key])

    dirty = st.session_state[dirty_key]

    ab1, ab2, ab3, ab4 = st.columns([2, 2, 2, 4])
    with ab1:
        save_clicked = st.button(
            "💾 Save", use_container_width=True,
            disabled=not dirty,
            type="primary" if dirty else "secondary",
            key=f"save_{file_path}",
        )
    with ab2:
        discard_clicked = st.button(
            "↩ Discard", use_container_width=True, disabled=not dirty,
            key=f"discard_{file_path}",
        )
    with ab3:
        reload_clicked = st.button(
            "🔄 Reload", use_container_width=True,
            help="Re-read from disk (discards unsaved edits)",
            key=f"reload_{file_path}",
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
        "Creates a new strategy folder with a Python file and a README. "
        "Customise both in the editor after creation."
        "</div>",
        unsafe_allow_html=True,
    )

    name = st.text_input("Strategy name", placeholder="e.g. Momentum Vol-Scaled")
    desc = st.text_input("Description",   placeholder="One-line summary of the strategy")

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
            st.error(f"strategies/{slug}/ already exists. Choose a different name.")
        else:
            target_dir.mkdir(parents=True)
            description = desc or "No description provided."
            (target_dir / f"{slug}.py").write_text(
                _PY_TEMPLATE.format(name=name, description=description),
                encoding="utf-8",
            )
            (target_dir / "README.md").write_text(
                _README_TEMPLATE.format(name=name, description=description),
                encoding="utf-8",
            )
            st.session_state["selected_strategy"] = str(target_dir)
            st.rerun()


# ── Results renderer ──────────────────────────────────────────────────────────

def _show_results(results_ph, console_ph, data: dict) -> None:
    console_lines = data.get("_console", [])

    if not data.get("ok"):
        with results_ph.container():
            err = data.get("error", "Unknown error")
            st.markdown(
                status_banner(f"Backtest failed — {err[:140]}", COLORS["negative"]),
                unsafe_allow_html=True,
            )
        with console_ph.container():
            if console_lines:
                with st.expander("Console output", expanded=True):
                    st.code("\n".join(console_lines), language="text")
        return

    metrics  = data.get("metrics", {})
    params   = data.get("params", {})
    equity   = data.get("equity", {})
    bench    = data.get("benchmark", {})
    strat_nm = data.get("strategy_name", "Strategy")
    sharpe   = float(metrics.get("sharpe") or 0)

    with results_ph.container():
        if sharpe >= 0.8:
            bc, bt = COLORS["positive"], f"Backtest complete — Sharpe {sharpe:.2f}"
        elif sharpe >= 0.4:
            bc, bt = COLORS["warning"],  f"Backtest complete — Sharpe {sharpe:.2f} (below target)"
        else:
            bc, bt = COLORS["negative"], f"Backtest complete — Sharpe {sharpe:.2f} (low)"
        st.markdown(status_banner(bt, bc), unsafe_allow_html=True)

        p_period = f"{metrics.get('start','?')} → {metrics.get('end','?')} ({metrics.get('years','?')}y)"
        p_cfg    = (
            f"Top-{params.get('top_n','?')} · "
            f"{params.get('weight_scheme','?')} weight · "
            f"{params.get('cost_bps','?')} bps"
        )
        st.markdown(
            f'<div style="color:{COLORS["neutral"]};font-size:0.76rem;margin-bottom:12px;">'
            f'<b style="color:{COLORS["text"]}">{strat_nm}</b> &nbsp;·&nbsp; '
            f'{p_period} &nbsp;·&nbsp; {p_cfg}</div>',
            unsafe_allow_html=True,
        )

        def _pct(v):  return f"{v:+.1%}" if isinstance(v, float) else "—"
        def _f(v):    return f"{v:.3f}"  if isinstance(v, float) else "—"
        def _cost(v): return f"${v:,.0f}" if isinstance(v, (int, float)) else "—"
        def _n(v):    return str(int(v)) if isinstance(v, (int, float)) else "—"

        kpis = [
            ("Total Return",     _pct(metrics.get("total_return")), None),
            ("CAGR",             _pct(metrics.get("cagr")),         None),
            ("Sharpe",           _f(sharpe), COLORS["positive"] if sharpe >= 0.8 else COLORS["warning"]),
            ("Sortino",          _f(metrics.get("sortino")),        None),
            ("Max Drawdown",     _pct(metrics.get("max_drawdown")), COLORS["negative"]),
            ("Calmar",           _f(metrics.get("calmar")),         None),
            ("Transaction Cost", _cost(metrics.get("total_cost")),  None),
            ("Trades",           _n(metrics.get("n_trades")),       None),
        ]
        for i in range(0, len(kpis), 2):
            c1, c2 = st.columns(2)
            for col, (label, val, accent) in zip([c1, c2], kpis[i:i + 2]):
                with col:
                    st.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

        if equity.get("dates"):
            eq_dates = pd.to_datetime(equity["dates"])
            eq_vals  = equity["values"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_dates, y=eq_vals,
                mode="lines", name=strat_nm,
                line=dict(color=COLORS["positive"], width=2),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
                hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
            ))
            if bench.get("dates") and eq_vals:
                b_dates = pd.to_datetime(bench["dates"])
                b_vals  = bench["values"]
                scale   = eq_vals[0] / b_vals[0] if b_vals and b_vals[0] else 1
                fig.add_trace(go.Scatter(
                    x=b_dates, y=[v * scale for v in b_vals],
                    mode="lines", name="SPY (benchmark)",
                    line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
                    hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
                ))
            apply_theme(fig, title="Equity Curve vs. Benchmark", height=280)
            st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")

    with console_ph.container():
        if console_lines:
            with st.expander("Console output", expanded=False):
                st.code("\n".join(console_lines), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(page_header(
    "Strategy Lab",
    "Edit strategies, configure parameters, and run backtests",
), unsafe_allow_html=True)

# ── Strategy selector bar ─────────────────────────────────────────────────────

st.markdown(section_label("Strategy"), unsafe_allow_html=True)

strategy_dirs    = _list_strategies()
strategy_options = _strategy_options(strategy_dirs)   # label -> dir Path

sel_col, new_col = st.columns([5, 1])

with new_col:
    if st.button("➕ New", use_container_width=True, help="Create a new strategy from template"):
        _new_strategy_dialog()

if not strategy_options:
    st.info("No strategies found in `strategies/`. Click **➕ New** to create your first one.")
    st.stop()

# Resolve which strategy is selected
_saved_path = st.session_state.get("selected_strategy")
default_idx = 0
labels      = list(strategy_options.keys())
if _saved_path:
    for i, d in enumerate(strategy_options.values()):
        if str(d) == _saved_path:
            default_idx = i
            break

with sel_col:
    selected_label = st.selectbox(
        "Strategy",
        options=labels,
        index=default_idx,
        label_visibility="collapsed",
    )

selected_dir = strategy_options[selected_label]
st.session_state["selected_strategy"] = str(selected_dir)

st.markdown(
    f'<div style="color:{COLORS["text_muted"]};font-size:0.72rem;margin:-4px 0 12px;">'
    f'strategies/{selected_dir.name}/</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Main layout ───────────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Code editor (with per-file tabs if multiple files exist)
# ══════════════════════════════════════════════════════════════════════════════

with left:
    st.markdown(section_label("Code Editor"), unsafe_allow_html=True)

    strat_files = _strategy_files(selected_dir)

    if not strat_files:
        st.warning("No files found in this strategy folder.")
    elif len(strat_files) == 1:
        _render_editor(strat_files[0])
    else:
        tab_labels = [f.name for f in strat_files]
        file_tabs  = st.tabs(tab_labels)
        for tab, file_path in zip(file_tabs, strat_files):
            with tab:
                _render_editor(file_path)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Backtest config + results
# ══════════════════════════════════════════════════════════════════════════════

with right:
    st.markdown(section_label("Backtest Configuration"), unsafe_allow_html=True)

    main_py = _main_py(selected_dir)

    # Load strategy defaults to pre-fill UI
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
            "Weighting",
            options=scheme_opts,
            index=scheme_opts.index(scheme_def) if scheme_def in scheme_opts else 0,
        )

    run_btn = st.button(
        "▶  Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=main_py is None,
        help=f"Backtest strategies/{selected_dir.name}/{main_py.name if main_py else ''}",
    )

    st.divider()

    results_ph = st.empty()
    console_ph = st.empty()

    result_key = f"lab_result_{selected_dir.name}"

    if result_key in st.session_state and not run_btn:
        _show_results(results_ph, console_ph, st.session_state[result_key])

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
            st.markdown(
                status_banner("Backtest running…", COLORS["info"], animate=True),
                unsafe_allow_html=True,
            )

        stdout_buf   = ""
        stderr_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=str(_ROOT),
            )
            stdout_buf, stderr_raw = proc.communicate(timeout=300)
            stderr_lines = [ln for ln in stderr_raw.splitlines() if ln.strip()]
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_buf, _ = proc.communicate()
            stderr_lines.append("ERROR: Backtest timed out after 5 minutes.")
        except Exception as exc:
            stderr_lines.append(f"ERROR launching subprocess: {exc}")

        try:
            result_data = json.loads(stdout_buf.strip()) if stdout_buf.strip() else {}
        except json.JSONDecodeError:
            result_data = {"ok": False, "error": f"Could not parse output:\n{stdout_buf[:400]}"}

        result_data["_console"] = stderr_lines
        st.session_state[result_key] = result_data
        _show_results(results_ph, console_ph, result_data)
