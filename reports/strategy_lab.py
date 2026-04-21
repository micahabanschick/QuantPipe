"""Strategy Lab — in-browser code editor + backtest runner."""

import json
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

# ── File catalogue ─────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent

_STRATEGY_FILES = {
    "Strategy — Momentum Signal": "signals/momentum.py",
    "Strategy — Canary Backtest":  "backtest/canary.py",
    "Config — Universe":           "config/universes.py",
    "Risk — Scenarios":            "risk/scenarios.py",
    "Pipeline — Generate Signals": "orchestration/generate_signals.py",
}

_ENGINE_FILES = {
    "Engine — Backtest Core":  "backtest/engine.py",
    "Engine — Optimizer":      "portfolio/optimizer.py",
    "Engine — Risk Engine":    "risk/engine.py",
}

_SEP = "─── Engine files (caution) ───"
_FILE_OPTIONS = list(_STRATEGY_FILES.keys()) + [_SEP] + list(_ENGINE_FILES.keys())
_ALL_FILES = {**_STRATEGY_FILES, **_ENGINE_FILES}

_ACE_THEME = "tomorrow_night"
_ACE_FONT_SIZE = 14


# ── Helper — render backtest results ──────────────────────────────────────────

def _show_results(results_ph, console_ph, data: dict) -> None:
    console_lines = data.get("_console", [])

    if not data.get("ok"):
        with results_ph.container():
            err = data.get("error", "Unknown error")
            st.markdown(
                status_banner(f"Backtest failed — {err[:120]}", COLORS["negative"]),
                unsafe_allow_html=True,
            )
        with console_ph.container():
            if console_lines:
                with st.expander("Console output", expanded=True):
                    st.code("\n".join(console_lines), language="text")
        return

    metrics = data.get("metrics", {})
    params  = data.get("params", {})
    equity  = data.get("equity", {})
    bench   = data.get("benchmark", {})
    sharpe  = metrics.get("sharpe", 0.0) or 0.0

    with results_ph.container():
        # Status banner
        if sharpe >= 0.8:
            bc, bt = COLORS["positive"], f"Backtest complete — Sharpe {sharpe:.2f}"
        elif sharpe >= 0.4:
            bc, bt = COLORS["warning"], f"Backtest complete — Sharpe {sharpe:.2f} (below target)"
        else:
            bc, bt = COLORS["negative"], f"Backtest complete — Sharpe {sharpe:.2f} (low)"
        st.markdown(status_banner(bt, bc), unsafe_allow_html=True)

        # Params summary
        p_period = f"{metrics.get('start','?')} → {metrics.get('end','?')} ({metrics.get('years','?')}y)"
        p_cfg    = (
            f"Top-{params.get('top_n','?')} · "
            f"{params.get('weight_scheme','?')} weight · "
            f"{params.get('cost_bps','?')} bps cost"
        )
        st.markdown(
            f'<div style="color:{COLORS["neutral"]};font-size:0.78rem;margin-bottom:12px;">'
            f'{p_period} &nbsp;|&nbsp; {p_cfg}</div>',
            unsafe_allow_html=True,
        )

        # KPI grid
        def _pct(v): return f"{v:+.1%}" if isinstance(v, float) else "—"
        def _f(v):   return f"{v:.3f}"  if isinstance(v, float) else "—"
        def _cost(v): return f"${v:,.0f}" if isinstance(v, (int, float)) else "—"
        def _n(v):   return str(int(v)) if isinstance(v, (int, float)) else "—"

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

        for row_start in range(0, len(kpis), 2):
            cols = st.columns(2)
            for col, (label, val, accent) in zip(cols, kpis[row_start:row_start + 2]):
                with col:
                    st.markdown(kpi_card(label, val, accent=accent), unsafe_allow_html=True)

        # Equity curve
        if equity.get("dates"):
            eq_dates = pd.to_datetime(equity["dates"])
            eq_vals  = equity["values"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_dates, y=eq_vals,
                mode="lines", name="Strategy",
                line=dict(color=COLORS["positive"], width=2),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
                hovertemplate="%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
            ))

            if bench.get("dates") and eq_vals:
                b_dates = pd.to_datetime(bench["dates"])
                b_vals  = bench["values"]
                scale   = eq_vals[0] / b_vals[0] if b_vals and b_vals[0] else 1
                b_scaled = [v * scale for v in b_vals]
                fig.add_trace(go.Scatter(
                    x=b_dates, y=b_scaled,
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


# ── Page header ────────────────────────────────────────────────────────────────

st.markdown(page_header(
    "Strategy Lab",
    "Edit strategy code and run backtests directly from the dashboard",
), unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Code editor
# ══════════════════════════════════════════════════════════════════════════════

with left:
    st.markdown(section_label("Code Editor"), unsafe_allow_html=True)

    selected_label = st.selectbox(
        "File",
        options=_FILE_OPTIONS,
        index=0,
        label_visibility="collapsed",
    )

    if selected_label == _SEP:
        st.info("Select a file from the list above to begin editing.")
        st.stop()

    is_engine = selected_label in _ENGINE_FILES
    rel_path  = _ALL_FILES[selected_label]
    abs_path  = _ROOT / rel_path

    if is_engine:
        st.markdown(
            status_banner(
                "Engine file — changes affect all backtests. Edit carefully.",
                COLORS["warning"],
            ),
            unsafe_allow_html=True,
        )

    # Session-state keys per file
    disk_key  = f"disk__{rel_path}"
    edit_key  = f"edit__{rel_path}"
    dirty_key = f"dirty_{rel_path}"

    def _load_from_disk() -> str:
        try:
            return abs_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"# File not found: {rel_path}\n"

    if disk_key not in st.session_state:
        st.session_state[disk_key]  = _load_from_disk()
        st.session_state[edit_key]  = st.session_state[disk_key]
        st.session_state[dirty_key] = False

    new_content = st_ace(
        value=st.session_state[edit_key],
        language="python",
        theme=_ACE_THEME,
        font_size=_ACE_FONT_SIZE,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=False,
        auto_update=True,
        min_lines=30,
        max_lines=55,
        key=f"ace_{rel_path}",
    )

    if new_content is not None:
        st.session_state[edit_key]  = new_content
        st.session_state[dirty_key] = (new_content != st.session_state[disk_key])

    dirty = st.session_state[dirty_key]

    # Action bar
    ab1, ab2, ab3, ab4 = st.columns([2, 2, 2, 4])

    with ab1:
        save_clicked = st.button(
            "💾 Save", use_container_width=True,
            disabled=not dirty,
            type="primary" if dirty else "secondary",
        )
    with ab2:
        discard_clicked = st.button(
            "↩ Discard", use_container_width=True,
            disabled=not dirty,
        )
    with ab3:
        reload_clicked = st.button(
            "🔄 Reload", use_container_width=True,
            help="Re-read from disk (discards unsaved edits)",
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
                f'<div style="padding:8px 0;color:{COLORS["positive"]};'
                f'font-size:0.80rem;">✓ Saved</div>',
                unsafe_allow_html=True,
            )

    if save_clicked:
        try:
            abs_path.write_text(st.session_state[edit_key], encoding="utf-8")
            st.session_state[disk_key]  = st.session_state[edit_key]
            st.session_state[dirty_key] = False
            st.success(f"Saved → {rel_path}")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    if discard_clicked:
        st.session_state[edit_key]  = st.session_state[disk_key]
        st.session_state[dirty_key] = False
        st.rerun()

    if reload_clicked:
        fresh = _load_from_disk()
        st.session_state[disk_key]  = fresh
        st.session_state[edit_key]  = fresh
        st.session_state[dirty_key] = False
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Backtest config + results
# ══════════════════════════════════════════════════════════════════════════════

with right:
    st.markdown(section_label("Backtest Configuration"), unsafe_allow_html=True)

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        lookback_years = st.number_input(
            "Lookback (years)", min_value=1, max_value=20, value=6, step=1,
        )
        cost_bps = st.number_input(
            "Cost (bps)", min_value=0.0, max_value=100.0,
            value=5.0, step=0.5, format="%.1f",
        )
    with cfg2:
        top_n = st.number_input(
            "Top-N positions", min_value=1, max_value=20, value=5, step=1,
        )
        weight_scheme = st.selectbox(
            "Weighting",
            options=["equal", "vol_scaled"],
            index=0,
        )

    run_btn = st.button(
        "▶  Run Backtest",
        type="primary",
        use_container_width=True,
        help="Executes tools/backtest_runner.py as a subprocess",
    )

    st.divider()

    results_ph = st.empty()
    console_ph = st.empty()

    # Restore prior results from session state on page reload
    if "lab_result" in st.session_state and not run_btn:
        _show_results(results_ph, console_ph, st.session_state["lab_result"])

    if run_btn:
        python_exe = sys.executable
        runner     = str(_ROOT / "tools" / "backtest_runner.py")
        cmd = [
            python_exe, runner,
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

        stdout_buf = ""
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
            stderr_lines = [l for l in stderr_raw.splitlines() if l.strip()]
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
        st.session_state["lab_result"] = result_data
        _show_results(results_ph, console_ph, result_data)
