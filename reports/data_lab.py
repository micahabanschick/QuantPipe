"""Data Lab — Alternative data ingestion, cleaning, and tradability analysis.

Pipeline position: Step 0.5 — before feature engineering.
Purpose: Evaluate whether a new non-price data source has enough
         predictive signal to justify building it into the formal pipeline.

Tabs:
    Source Registry  — scaffold for future automated connectors (coming soon)
    Ingest & Clean   — upload CSV/JSON, map columns, clean, save to data/alt/
    Tradability      — IC at multiple lags, rolling IC, scatter, verdict
"""

from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
import streamlit as st

from config.settings import DATA_DIR, FRED_API_KEY
from data_adapters.fred_adapter import FREDAdapter, POPULAR_SERIES
from storage.parquet_store import load_bars
from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, apply_subplot_theme,
    kpi_card, section_label, badge, page_header, status_banner,
)

st.markdown(CSS, unsafe_allow_html=True)

ALT_DIR = DATA_DIR / "alt"
ALT_DIR.mkdir(parents=True, exist_ok=True)

# ── Page header ────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Data Lab",
        "Ingest and clean alternative data sources, then test their predictive power before building them into the pipeline.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

tab_reg, tab_ingest, tab_trade = st.tabs(
    ["  Source Registry  ", "  Ingest & Clean  ", "  Tradability Check  "]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SOURCE REGISTRY (scaffolded)
# ══════════════════════════════════════════════════════════════════════════════

with tab_reg:
    st.markdown(section_label("Live Connectors"), unsafe_allow_html=True)

    # ── FRED — Live ────────────────────────────────────────────────────────────
    _fred_status = "LIVE" if FRED_API_KEY else "KEY MISSING"
    _fred_color  = COLORS["green"] if FRED_API_KEY else COLORS["negative"]
    _fred_bg     = "rgba(0,230,118,0.08)" if FRED_API_KEY else "rgba(255,77,77,0.08)"
    _sample_series = list(POPULAR_SERIES.items())[:6]
    _series_html = "".join(
        f'<li style="color:{COLORS["text_muted"]};font-size:0.76rem;">'
        f'<code style="color:{COLORS["gold"]};font-size:0.72rem;">{sid}</code>'
        f' — {desc}</li>'
        for sid, desc in _sample_series
    )
    st.markdown(f"""
<div style="background:{_fred_bg};
            border:1px solid {_fred_color}44;
            border-left:3px solid {_fred_color};
            border-radius:8px;padding:14px 16px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="font-size:1.1rem;">📡</span>
    <span style="color:{COLORS['text']};font-size:0.92rem;font-weight:700;">
      FRED — Federal Reserve Economic Data
    </span>
    <span style="background:{_fred_color}22;color:{_fred_color};
                 font-size:0.62rem;font-weight:700;letter-spacing:0.08em;
                 padding:2px 8px;border-radius:3px;border:1px solid {_fred_color}55;">
      {_fred_status}
    </span>
  </div>
  <p style="color:{COLORS['neutral']};font-size:0.78rem;margin:0 0 8px;line-height:1.5;">
    Free macro and financial time series from the St. Louis Fed.
    {len(POPULAR_SERIES)} curated series available — pull any via the
    <b>Ingest &amp; Clean</b> tab, or enter any custom FRED series ID.
  </p>
  <ul style="margin:0;padding-left:16px;">{_series_html}</ul>
  <p style="color:{COLORS['text_muted']};font-size:0.72rem;margin:8px 0 0;">
    + {len(POPULAR_SERIES) - 6} more curated series · 800,000+ series available via search
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown(section_label("Planned Connectors"), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{COLORS["neutral"]};font-size:0.84rem;margin-bottom:18px;">'
        "These connectors will be built out as each data source is validated in the "
        "Tradability Check tab. Use Ingest &amp; Clean to manually upload and test a "
        "source before committing to a full automated connector."
        "</p>",
        unsafe_allow_html=True,
    )

    _sources = [
        ("🌐", "Web Scrapers",
         "Job postings (LinkedIn/Indeed), Glassdoor reviews, "
         "regulatory filings, product pricing.",
         ["Job openings per ticker", "Employee sentiment score",
          "Product price index", "Review volume & rating"]),
        ("📡", "Other API Connectors",
         "Additional macro and alternative data APIs: Quandl, "
         "Alpha Vantage, World Bank, BLS.",
         ["Harvest cycle indices", "Credit card spend by sector",
          "World Bank development data", "BLS detailed labour stats"]),
        ("📂", "File Watchers",
         "Scheduled CSV/JSON drops from data vendors, "
         "internal research exports, survey outputs.",
         ["Vendor data dumps", "Internal model outputs",
          "Survey results", "Earnings call NLP scores"]),
        ("🗄️", "Database Connectors",
         "Direct SQL/NoSQL connections to proprietary data "
         "stores, data warehouses, or third-party platforms.",
         ["PostgreSQL / Snowflake", "MongoDB collections",
          "REST + webhook ingestion", "S3 / GCS data lake"]),
        ("📰", "News & Sentiment",
         "Financial news APIs, Reddit/Twitter sentiment, "
         "earnings call transcripts, SEC 8-K filings.",
         ["FinBERT sentiment scores", "News volume by ticker",
          "Social media momentum", "Insider filing tone"]),
        ("🌦️", "Environmental & Macro",
         "Weather patterns, commodity supply chain data, "
         "shipping indices, energy demand cycles.",
         ["Weather anomaly index", "Baltic Dry Index",
          "Natural gas demand", "Agricultural yield forecasts"]),
    ]

    cols = st.columns(2)
    for i, (icon, name, desc, examples) in enumerate(_sources):
        with cols[i % 2]:
            examples_html = "".join(
                f'<li style="color:{COLORS["text_muted"]};font-size:0.76rem;">{e}</li>'
                for e in examples
            )
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#1E2848,#0F1528);
            border:1px solid rgba(201,162,39,0.18);
            border-left:3px solid {COLORS['gold_dim']};
            border-radius:8px;padding:14px 16px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="font-size:1.1rem;">{icon}</span>
    <span style="color:{COLORS['text']};font-size:0.88rem;font-weight:700;">{name}</span>
    <span style="background:rgba(201,162,39,0.12);color:{COLORS['gold_dim']};
                 font-size:0.62rem;font-weight:700;letter-spacing:0.08em;
                 padding:2px 8px;border-radius:3px;border:1px solid rgba(201,162,39,0.25);">
      COMING SOON
    </span>
  </div>
  <p style="color:{COLORS['neutral']};font-size:0.78rem;margin:0 0 8px;line-height:1.5;">
    {desc}
  </p>
  <ul style="margin:0;padding-left:16px;">{examples_html}</ul>
</div>
""", unsafe_allow_html=True)

    st.info(
        "To request a connector or contribute one, open a GitHub issue or add the "
        "adapter to `data_adapters/` following the `BaseAdapter` protocol.",
        icon="💡",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INGEST & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

with tab_ingest:

    # ── Source selector ────────────────────────────────────────────────────────
    _fred_available = bool(FRED_API_KEY)
    _src_options = ["📡 FRED — Federal Reserve", "📁 Upload File (CSV / JSON)"]
    if not _fred_available:
        _src_options = ["📁 Upload File (CSV / JSON)", "📡 FRED — Federal Reserve (key missing)"]

    ingest_source = st.radio(
        "Data source", _src_options, horizontal=True, key="ingest_src",
    )
    _is_fred = ingest_source.startswith("📡 FRED") and _fred_available

    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FRED PULL PATH
    # ══════════════════════════════════════════════════════════════════════════
    if _is_fred:
        fred = FREDAdapter(FRED_API_KEY)

        st.markdown(section_label("Select FRED Series"), unsafe_allow_html=True)

        fi1, fi2 = st.columns([2, 2])
        with fi1:
            popular_choice = st.selectbox(
                "Popular series",
                ["— search or pick below —"] + list(POPULAR_SERIES.keys()),
                format_func=lambda k: k if k.startswith("—") else f"{k} — {POPULAR_SERIES.get(k, '')}",
                key="fred_popular",
            )
        with fi2:
            custom_id = st.text_input(
                "Or enter any FRED series ID",
                placeholder="e.g. MORTGAGE30US",
                key="fred_custom",
            )

        series_id = (
            custom_id.strip().upper()
            if custom_id.strip()
            else (popular_choice if not popular_choice.startswith("—") else None)
        )

        # Search box
        search_q = st.text_input("🔍 Search FRED catalogue", placeholder="inflation, housing…", key="fred_search")
        if search_q:
            with st.spinner("Searching FRED…"):
                results = fred.search_series(search_q, limit=10)
            if results:
                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True, hide_index=True,
                )
                st.caption("Copy a series ID above into the 'Enter any FRED series ID' box.")
            else:
                st.info("No results found.")

        if not series_id:
            st.info("Pick a popular series or enter a custom series ID above.")
            st.stop()

        # Show series metadata
        with st.spinner(f"Fetching metadata for {series_id}…"):
            info = fred.get_series_info(series_id)
        st.markdown(
            f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
            f'border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.80rem;">'
            f'<b style="color:{COLORS["gold"]};">{info["id"]}</b> — '
            f'<span style="color:{COLORS["text"]};">{info["title"]}</span><br>'
            f'<span style="color:{COLORS["text_muted"]};">Units: {info["units"]} · '
            f'Frequency: {info["frequency"]} · Last updated: {info["last_updated"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        fd1, fd2 = st.columns(2)
        with fd1:
            fred_start = st.date_input("Start date", value=date(2015, 1, 1), key="fred_start")
        with fd2:
            fred_end   = st.date_input("End date",   value=date.today(),      key="fred_end")

        source_name = st.text_input(
            "Save as", value=f"fred_{series_id.lower()}", key="fred_savename"
        )

        if st.button("⬇ Pull from FRED", key="fred_pull"):
            with st.spinner(f"Pulling {series_id} from {fred_start} to {fred_end}…"):
                try:
                    raw_pl = fred.get_series(series_id, start=str(fred_start), end=str(fred_end))
                    if raw_pl.is_empty():
                        st.error("No data returned. Check the series ID and date range.")
                        st.stop()
                except Exception as exc:
                    st.error(f"FRED API error: {exc}")
                    st.stop()

            raw_pd = raw_pl.to_pandas()
            st.success(f"Pulled {len(raw_pd):,} rows for {series_id}")

            # Preview
            st.markdown(section_label("Preview"), unsafe_allow_html=True)
            n_null = raw_pd[series_id].isnull().sum()
            p1, p2, p3 = st.columns(3)
            p1.metric("Rows",        f"{len(raw_pd):,}")
            p2.metric("Null values", f"{n_null:,}")
            p3.metric("Date range",  f"{raw_pd['date'].min().date()} → {raw_pd['date'].max().date()}")

            st.dataframe(raw_pd.head(10), use_container_width=True, hide_index=True)

            fig_fred = go.Figure(go.Scatter(
                x=raw_pd["date"], y=raw_pd[series_id],
                mode="lines", line=dict(color=COLORS["gold"], width=1.8),
                name=series_id,
                hovertemplate="%{x}<br>%{y}<extra></extra>",
            ))
            apply_theme(fig_fred, title=f"{info['title']} ({info['units']})", height=260)
            st.plotly_chart(fig_fred, use_container_width=True, config=PLOTLY_CONFIG)

            # Save
            out_path = ALT_DIR / f"{source_name}.parquet"
            raw_pl.write_parquet(out_path)
            st.success(f"Saved to `data/alt/{source_name}.parquet`")
            st.info("Open the **Tradability Check** tab to analyse this signal.")

        st.stop()   # don't render the upload section when FRED is selected

    # ── Upload path ─────────────────────────────────────────────────────────────
    st.markdown(section_label("Upload Alternative Data"), unsafe_allow_html=True)
    st.caption(
        "CSV or JSON. Must contain a date column and at least one numeric signal column. "
        "Panel data (date × symbol) is supported — map the symbol column below."
    )

    uploaded = st.file_uploader(
        "Drop a CSV or JSON file",
        type=["csv", "json"],
        key="dlab_upload",
    )

    if uploaded is None:
        # Show existing saved files
        saved = sorted(ALT_DIR.glob("*.parquet"))
        if saved:
            st.markdown(section_label("Saved Alt Data Files"), unsafe_allow_html=True)
            for f in saved:
                size = f.stat().st_size / 1024
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(
                    f'<span style="color:{COLORS["text"]};font-size:0.84rem;">'
                    f'📄 {f.stem}</span>',
                    unsafe_allow_html=True,
                )
                c2.markdown(
                    f'<span style="color:{COLORS["text_muted"]};font-size:0.78rem;">'
                    f'{size:.1f} KB</span>',
                    unsafe_allow_html=True,
                )
                if c3.button("🗑 Delete", key=f"del_{f.stem}"):
                    f.unlink()
                    st.rerun()
        else:
            st.info("No alt data files saved yet. Upload a file above to get started.")
        st.stop()

    # ── Parse ──────────────────────────────────────────────────────────────────
    try:
        raw_text = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".json"):
            raw_df = pd.read_json(StringIO(raw_text))
        else:
            raw_df = pd.read_csv(StringIO(raw_text))
    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        st.stop()

    st.markdown(section_label(f"Raw Preview — {uploaded.name}"), unsafe_allow_html=True)
    st.markdown(
        f'<span style="color:{COLORS["neutral"]};font-size:0.78rem;">'
        f'{len(raw_df):,} rows · {len(raw_df.columns)} columns</span>',
        unsafe_allow_html=True,
    )
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

    # ── Column mapping ─────────────────────────────────────────────────────────
    st.markdown(section_label("Column Mapping"), unsafe_allow_html=True)
    all_cols   = list(raw_df.columns)
    str_cols   = [c for c in all_cols if raw_df[c].dtype == object]
    num_cols   = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw_df[c])]

    # Auto-detect date column
    _date_guess = next(
        (c for c in all_cols if "date" in c.lower() or "time" in c.lower()), all_cols[0]
    )
    _sym_guess  = next(
        (c for c in str_cols if any(k in c.lower() for k in ["ticker", "symbol", "sym"])),
        "— none —",
    )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        date_col = st.selectbox("Date column", all_cols,
                                index=all_cols.index(_date_guess), key="dlab_datecol")
    with mc2:
        sym_options = ["— none —"] + str_cols
        sym_col = st.selectbox(
            "Symbol column (panel data only)",
            sym_options,
            index=sym_options.index(_sym_guess) if _sym_guess in sym_options else 0,
            key="dlab_symcol",
        )
    with mc3:
        signal_cols = st.multiselect(
            "Signal column(s)",
            [c for c in num_cols],
            default=num_cols[:1] if num_cols else [],
            key="dlab_sigcols",
        )

    source_name = st.text_input(
        "Source name (used as filename)",
        value=uploaded.name.rsplit(".", 1)[0].replace(" ", "_").lower(),
        key="dlab_srcname",
    )

    if not signal_cols:
        st.warning("Select at least one signal column to continue.")
        st.stop()

    # ── Cleaning controls ──────────────────────────────────────────────────────
    st.markdown(section_label("Cleaning Options"), unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        fill_method = st.selectbox(
            "Fill nulls",
            ["forward-fill", "backfill", "interpolate (linear)", "none"],
            key="dlab_fill",
        )
    with cc2:
        outlier_method = st.selectbox(
            "Outlier handling",
            ["winsorize (1–99%)", "z-score clip (±3σ)", "none"],
            key="dlab_outlier",
        )
    with cc3:
        freq = st.selectbox(
            "Resample to frequency",
            ["none (keep as-is)", "daily (B)", "weekly (W-FRI)", "monthly (MS)"],
            key="dlab_freq",
        )

    # ── Apply cleaning ─────────────────────────────────────────────────────────
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Parse date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        # Keep relevant columns
        keep = [date_col] + ([sym_col] if sym_col != "— none —" else []) + signal_cols
        df = df[keep].copy()

        # Fill nulls
        if fill_method == "forward-fill":
            df[signal_cols] = df[signal_cols].ffill()
        elif fill_method == "backfill":
            df[signal_cols] = df[signal_cols].bfill()
        elif fill_method == "interpolate (linear)":
            df[signal_cols] = df[signal_cols].interpolate(method="linear", limit_direction="both")

        # Outlier clipping
        if outlier_method == "winsorize (1–99%)":
            for c in signal_cols:
                lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
                df[c] = df[c].clip(lo, hi)
        elif outlier_method == "z-score clip (±3σ)":
            for c in signal_cols:
                mu, sd = df[c].mean(), df[c].std()
                if sd > 0:
                    df[c] = df[c].clip(mu - 3 * sd, mu + 3 * sd)

        # Resample
        if freq != "none (keep as-is)":
            rule = {"daily (B)": "B", "weekly (W-FRI)": "W-FRI", "monthly (MS)": "MS"}[freq]
            df = df.set_index(date_col)
            if sym_col != "— none —":
                df = df.groupby(sym_col).resample(rule)[signal_cols].last().reset_index()
            else:
                df = df.resample(rule)[signal_cols].last().reset_index()
                df = df.rename(columns={"index": date_col})

        return df

    clean_df = _clean(raw_df)

    # ── Cleaned preview ────────────────────────────────────────────────────────
    st.markdown(section_label("Cleaned Preview"), unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Rows (raw)",     f"{len(raw_df):,}")
    s2.metric("Rows (cleaned)", f"{len(clean_df):,}")
    s3.metric("Null values",    f"{clean_df[signal_cols].isnull().sum().sum():,}")

    st.dataframe(clean_df.head(20), use_container_width=True, hide_index=True)

    # Signal preview chart
    if len(clean_df) > 1 and date_col in clean_df.columns:
        fig_sig = go.Figure()
        for sc in signal_cols:
            if sym_col != "— none —":
                for sym, grp in clean_df.groupby(sym_col):
                    fig_sig.add_trace(go.Scatter(
                        x=grp[date_col], y=grp[sc],
                        name=f"{sc} / {sym}", mode="lines", line=dict(width=1.5),
                    ))
            else:
                fig_sig.add_trace(go.Scatter(
                    x=clean_df[date_col], y=clean_df[sc],
                    name=sc, mode="lines", line=dict(width=2),
                ))
        apply_theme(fig_sig, title="Cleaned Signal Time Series", height=280)
        st.plotly_chart(fig_sig, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Save ───────────────────────────────────────────────────────────────────
    if st.button("💾 Save to Data Lab", key="dlab_save", use_container_width=False):
        out_path = ALT_DIR / f"{source_name}.parquet"
        pl.from_pandas(clean_df.rename(columns={date_col: "date"})).write_parquet(out_path)
        st.success(f"Saved to `data/alt/{source_name}.parquet` ({len(clean_df):,} rows)")
        st.info("Open the **Tradability Check** tab to analyse this signal.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADABILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

with tab_trade:

    saved_files = sorted(ALT_DIR.glob("*.parquet"))
    if not saved_files:
        st.info(
            "No saved alt data found. Upload and save a file in **Ingest & Clean** first.",
            icon="📂",
        )
        st.stop()

    # ── Controls ────────────────────────────────────────────────────────────────
    tc1, tc2, tc3, tc4 = st.columns([2, 1.5, 1.5, 1])
    with tc1:
        source_file = st.selectbox(
            "Alt data source",
            [f.stem for f in saved_files],
            key="tc_source",
        )
    with tc2:
        equity_sym = st.selectbox(
            "Equity target",
            ["SPY", "QQQ", "IWM", "Equal-weight universe"],
            key="tc_equity",
        )
    with tc3:
        signal_col_tc = st.selectbox(
            "Signal column",
            options=[],          # populated after loading
            key="tc_sigcol",
        )
    with tc4:
        st.markdown("<div style='height:28px'/>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Analysis", key="tc_run", use_container_width=True)

    # Load alt data
    alt_path = ALT_DIR / f"{source_file}.parquet"
    try:
        alt_pl = pl.read_parquet(alt_path)
        alt_pd = alt_pl.to_pandas()
        if "date" in alt_pd.columns:
            alt_pd["date"] = pd.to_datetime(alt_pd["date"])
            alt_pd = alt_pd.sort_values("date").set_index("date")
        num_alt = [c for c in alt_pd.columns if pd.api.types.is_numeric_dtype(alt_pd[c])]
    except Exception as exc:
        st.error(f"Could not load {source_file}: {exc}")
        st.stop()

    # Update signal column selectbox options
    if num_alt:
        signal_col_tc = st.session_state.get("tc_sigcol", num_alt[0])
        if signal_col_tc not in num_alt:
            signal_col_tc = num_alt[0]
        # Re-render with options
        with tc3:
            signal_col_tc = st.selectbox(
                "Signal column",
                num_alt,
                index=num_alt.index(signal_col_tc),
                key="tc_sigcol2",
            )

    if not run_btn:
        st.markdown(
            section_label("Source Preview"), unsafe_allow_html=True
        )
        st.dataframe(alt_pd.reset_index().head(10), use_container_width=True, hide_index=True)
        st.caption(f"{len(alt_pd):,} rows · columns: {', '.join(alt_pd.columns.tolist())}")
        st.stop()

    # ── Load equity returns ────────────────────────────────────────────────────
    start_d = alt_pd.index.min().date() if hasattr(alt_pd.index.min(), "date") else date(2019, 1, 1)
    end_d   = alt_pd.index.max().date() if hasattr(alt_pd.index.max(), "date") else date.today()

    with st.spinner("Loading equity prices…"):
        if equity_sym == "Equal-weight universe":
            from storage.universe import universe_as_of_date
            syms = universe_as_of_date("equity", end_d)[:10]
            bars = load_bars(syms, start_d, end_d, "equity")
            price_col = "adj_close" if "adj_close" in bars.columns else "close"
            eq_prices = (
                bars.to_pandas()
                .pivot(index="date", columns="symbol", values=price_col)
                .mean(axis=1)
            )
        else:
            bars = load_bars([equity_sym], start_d, end_d, "equity")
            price_col = "adj_close" if "adj_close" in bars.columns else "close"
            eq_prices = bars.to_pandas().set_index("date")[price_col]

    eq_prices.index = pd.to_datetime(eq_prices.index)

    # ── Compute IC at multiple lags ────────────────────────────────────────────
    signal = alt_pd[signal_col_tc].dropna()
    lags   = [1, 5, 21, 63]
    lag_labels = ["1d", "5d", "21d", "63d"]
    lag_results = []

    for lag in lags:
        fwd_ret = eq_prices.pct_change(lag).shift(-lag)
        aligned = pd.concat([signal.rename("signal"), fwd_ret.rename("fwd_ret")], axis=1).dropna()
        if len(aligned) < 10:
            lag_results.append({"lag": f"{lag}d", "IC": np.nan, "p_value": np.nan,
                                 "n_obs": len(aligned), "IR": np.nan})
            continue
        ic, pval = spearmanr(aligned["signal"], aligned["fwd_ret"])
        lag_results.append({
            "lag":     f"{lag}d",
            "IC":      round(float(ic), 4),
            "p_value": round(float(pval), 4),
            "n_obs":   len(aligned),
            "IR":      round(float(ic) / aligned["fwd_ret"].std(), 4)
                       if aligned["fwd_ret"].std() > 0 else np.nan,
        })

    lag_df = pd.DataFrame(lag_results)

    # ── Rolling IC (21-period window on 21d forward returns) ──────────────────
    roll_ic = []
    fwd21 = eq_prices.pct_change(21).shift(-21)
    aligned21 = pd.concat([signal.rename("signal"), fwd21.rename("fwd21")], axis=1).dropna()
    window = max(21, len(aligned21) // 10)
    for i in range(window, len(aligned21)):
        chunk = aligned21.iloc[i - window: i]
        ic_r, _ = spearmanr(chunk["signal"], chunk["fwd21"])
        roll_ic.append({"date": aligned21.index[i], "rolling_ic": float(ic_r)})
    roll_ic_df = pd.DataFrame(roll_ic)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    best_lag = lag_df.loc[lag_df["IC"].abs().idxmax()] if not lag_df["IC"].isna().all() else None

    st.markdown(section_label("Tradability Summary"), unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    if best_lag is not None:
        best_ic   = best_lag["IC"]
        best_pval = best_lag["p_value"]
        verdict   = (
            "Strong signal" if abs(best_ic) > 0.05 and best_pval < 0.05
            else "Weak signal" if abs(best_ic) > 0.02
            else "No signal"
        )
        verdict_color = (
            COLORS["positive"] if "Strong" in verdict
            else COLORS["warning"] if "Weak" in verdict
            else COLORS["negative"]
        )
        k1.markdown(kpi_card("Best IC", f"{best_ic:+.4f}", accent=verdict_color),
                    unsafe_allow_html=True)
        k2.markdown(kpi_card("Best Lag", best_lag["lag"], accent=COLORS["gold"]),
                    unsafe_allow_html=True)
        k3.markdown(kpi_card("p-value", f"{best_pval:.4f}",
                              accent=COLORS["positive"] if best_pval < 0.05 else COLORS["negative"]),
                    unsafe_allow_html=True)
        k4.markdown(kpi_card("Verdict", verdict, accent=verdict_color),
                    unsafe_allow_html=True)
        st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
        st.markdown(
            status_banner(
                f"Signal: {verdict} — Best IC {best_ic:+.4f} at {best_lag['lag']} "
                f"forward return (p={best_pval:.4f}, n={int(best_lag['n_obs'])} obs)",
                color=verdict_color,
            ),
            unsafe_allow_html=True,
        )

    # ── IC lag scan bar chart ─────────────────────────────────────────────────
    st.markdown(section_label("IC at Multiple Forward Return Horizons"),
                unsafe_allow_html=True)

    bar_colors = [
        COLORS["positive"] if v > 0.05
        else COLORS["negative"] if v < -0.05
        else COLORS["gold_dim"]
        for v in lag_df["IC"].fillna(0)
    ]
    fig_lag = go.Figure()
    fig_lag.add_trace(go.Bar(
        x=lag_df["lag"], y=lag_df["IC"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:+.4f}" if not np.isnan(v) else "n/a" for v in lag_df["IC"]],
        textposition="outside",
        textfont=dict(color=COLORS["neutral"], size=11),
        hovertemplate="<b>%{x}</b><br>IC: %{y:.4f}<extra></extra>",
    ))
    fig_lag.add_hline(y=0.05,  line=dict(color=COLORS["positive"], width=1, dash="dot"),
                      annotation_text="Strong (+0.05)", annotation_font_color=COLORS["positive"])
    fig_lag.add_hline(y=-0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
    fig_lag.add_hline(y=0,     line=dict(color=COLORS["neutral"], width=1))
    apply_theme(fig_lag, title="Spearman IC by Forward Return Horizon", height=280)
    fig_lag.update_layout(showlegend=False, xaxis_title="Forward return lag",
                          yaxis_title="IC (Spearman ρ)")
    st.plotly_chart(fig_lag, use_container_width=True, config=PLOTLY_CONFIG)

    # ── IC table ──────────────────────────────────────────────────────────────
    st.dataframe(
        lag_df.style.format({"IC": "{:+.4f}", "p_value": "{:.4f}", "IR": "{:+.4f}"})
        .applymap(
            lambda v: f"color:{COLORS['positive']}" if isinstance(v, float) and abs(v) > 0.05
            else f"color:{COLORS['negative']}" if isinstance(v, float) and not np.isnan(v)
            else "",
            subset=["IC"],
        ),
        use_container_width=True, hide_index=True,
    )

    # ── Rolling IC ────────────────────────────────────────────────────────────
    if not roll_ic_df.empty:
        st.markdown(section_label(f"Rolling IC — {window}-period window (21d fwd returns)"),
                    unsafe_allow_html=True)
        fig_roll = go.Figure()
        fig_roll.add_hline(y=0.05,  line=dict(color=COLORS["positive"], width=1, dash="dot"))
        fig_roll.add_hline(y=-0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
        fig_roll.add_hline(y=0,     line=dict(color=COLORS["neutral"], width=1))
        fig_roll.add_trace(go.Scatter(
            x=roll_ic_df["date"], y=roll_ic_df["rolling_ic"],
            mode="lines", line=dict(color=COLORS["gold"], width=1.8),
            name="Rolling IC",
            hovertemplate="<b>%{x}</b><br>IC: %{y:.4f}<extra></extra>",
        ))
        fig_roll.add_trace(go.Scatter(
            x=roll_ic_df["date"],
            y=roll_ic_df["rolling_ic"].rolling(5, min_periods=1).mean(),
            mode="lines", line=dict(color=COLORS["green"], width=1, dash="dot"),
            name="5-period MA",
        ))
        apply_theme(fig_roll, title="Rolling IC over time", height=280, legend_inside=True)
        st.plotly_chart(fig_roll, use_container_width=True, config=PLOTLY_CONFIG)

    # ── Scatter ───────────────────────────────────────────────────────────────
    best_lag_days = int(best_lag["lag"].replace("d", "")) if best_lag is not None else 21
    fwd_best = eq_prices.pct_change(best_lag_days).shift(-best_lag_days)
    scatter_df = pd.concat(
        [signal.rename("signal"), fwd_best.rename("fwd_ret")], axis=1
    ).dropna()

    if len(scatter_df) > 5:
        st.markdown(
            section_label(f"Signal vs {best_lag['lag']} Forward Return — Scatter"),
            unsafe_allow_html=True,
        )
        m, b = np.polyfit(scatter_df["signal"], scatter_df["fwd_ret"], 1)
        x_line = np.linspace(scatter_df["signal"].min(), scatter_df["signal"].max(), 50)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=scatter_df["signal"], y=scatter_df["fwd_ret"],
            mode="markers",
            marker=dict(color=COLORS["gold"], size=5, opacity=0.6,
                        line=dict(width=0)),
            name="Observations",
            hovertemplate="Signal: %{x:.3f}<br>Fwd ret: %{y:.3%}<extra></extra>",
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", line=dict(color=COLORS["green"], width=2),
            name="OLS fit",
        ))
        apply_theme(fig_sc, title=f"Signal vs {best_lag['lag']} forward return",
                    height=320, legend_inside=True)
        fig_sc.update_layout(
            xaxis_title=f"Signal ({signal_col_tc})",
            yaxis_title=f"{best_lag['lag']} forward return",
            yaxis=dict(tickformat=".1%"),
        )
        st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)
