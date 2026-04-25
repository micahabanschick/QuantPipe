"""Data Lab — Alternative data ingestion, cleaning, and tradability analysis.

Pipeline position: Step 0.5 — before feature engineering.
Purpose: Evaluate whether a new non-price data source has enough
         predictive signal to justify building it into the formal pipeline.

Tabs:
    Source Registry  — connector roadmap (FRED live, others coming soon)
    Ingest & Clean   — pull from FRED or upload CSV/JSON, clean, save
    Tradability      — Spearman IC at multiple lags, rolling IC, scatter
"""

import json
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from scipy.stats import spearmanr
import streamlit as st

from config.settings import DATA_DIR, FRED_API_KEY
from data_adapters.fred_adapter import FREDAdapter, POPULAR_SERIES
from storage.parquet_store import load_bars
from reports._theme import (
    CSS, COLORS, PLOTLY_CONFIG,
    apply_theme, kpi_card, section_label, badge, page_header, status_banner,
)

st.markdown(CSS, unsafe_allow_html=True)

ALT_DIR = DATA_DIR / "alt"
ALT_DIR.mkdir(parents=True, exist_ok=True)
_META_FILE = ALT_DIR / ".meta.json"


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _load_meta() -> dict:
    if _META_FILE.exists():
        try:
            return json.loads(_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_meta(name: str, record: dict) -> None:
    meta = _load_meta()
    meta[name] = record
    _META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── Page header ────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Data Lab",
        "Ingest and clean alternative data sources, then test their predictive"
        " power before building them into the pipeline.",
        date.today().strftime("%B %d, %Y"),
    ),
    unsafe_allow_html=True,
)

tab_reg, tab_ingest, tab_trade = st.tabs(
    ["  Source Registry  ", "  Ingest & Clean  ", "  Tradability Check  "]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SOURCE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

with tab_reg:
    _fred_live = bool(FRED_API_KEY)
    _fc        = COLORS["green"] if _fred_live else COLORS["negative"]
    _fbg       = "rgba(0,230,118,0.08)" if _fred_live else "rgba(255,77,77,0.08)"
    _flabel    = "LIVE" if _fred_live else "KEY MISSING"
    _fsamples  = "".join(
        f'<li style="color:{COLORS["text_muted"]};font-size:0.76rem;">'
        f'<code style="color:{COLORS["gold"]};font-size:0.72rem;">{sid}</code>'
        f" — {desc}</li>"
        for sid, desc in list(POPULAR_SERIES.items())[:6]
    )
    st.markdown(section_label("Live Connectors"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:{_fbg};border:1px solid {_fc}44;border-left:3px solid {_fc};
            border-radius:8px;padding:14px 16px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="font-size:1.1rem;">&#x1F4E1;</span>
    <span style="color:{COLORS['text']};font-size:0.92rem;font-weight:700;">FRED — Federal Reserve Economic Data</span>
    <span style="background:{_fc}22;color:{_fc};font-size:0.62rem;font-weight:700;
                 letter-spacing:0.08em;padding:2px 8px;border-radius:3px;
                 border:1px solid {_fc}55;">{_flabel}</span>
  </div>
  <p style="color:{COLORS['neutral']};font-size:0.78rem;margin:0 0 8px;line-height:1.5;">
    {len(POPULAR_SERIES)} curated macro &amp; financial series.
    Pull any via <b>Ingest &amp; Clean</b>, or enter any custom FRED series ID.
  </p>
  <ul style="margin:0;padding-left:16px;">{_fsamples}</ul>
  <p style="color:{COLORS['text_muted']};font-size:0.72rem;margin:8px 0 0;">
    + {len(POPULAR_SERIES)-6} more curated &middot; 800,000+ series via search
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown(section_label("Planned Connectors"), unsafe_allow_html=True)
    st.caption("Validate each source in Tradability Check before building a full connector.")
    _planned = [
        ("Web Scrapers", "Job postings, Glassdoor reviews, regulatory filings.",
         ["Job openings per ticker", "Employee sentiment score"]),
        ("Other API Connectors", "Quandl, Alpha Vantage, World Bank, BLS.",
         ["Harvest cycle indices", "Credit card spend by sector"]),
        ("File Watchers", "Scheduled CSV/JSON drops from vendors.",
         ["Vendor data dumps", "Earnings call NLP scores"]),
        ("Database Connectors", "PostgreSQL, Snowflake, MongoDB, S3.",
         ["SQL data warehouse", "REST + webhook"]),
        ("News & Sentiment", "FinBERT, Reddit/Twitter, SEC 8-K.",
         ["Sentiment scores", "News volume by ticker"]),
        ("Environmental & Macro", "Weather, shipping indices, commodities.",
         ["Weather anomaly index", "Baltic Dry Index"]),
    ]
    cols2 = st.columns(2)
    for i, (name, desc, examples) in enumerate(_planned):
        ex_html = "".join(
            f'<li style="color:{COLORS["text_muted"]};font-size:0.76rem;">{e}</li>'
            for e in examples
        )
        with cols2[i % 2]:
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#1E2848,#0F1528);
            border:1px solid rgba(201,162,39,0.18);border-left:3px solid {COLORS['gold_dim']};
            border-radius:8px;padding:14px 16px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <span style="color:{COLORS['text']};font-size:0.88rem;font-weight:700;">{name}</span>
    <span style="background:rgba(201,162,39,0.12);color:{COLORS['gold_dim']};
                 font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:3px;
                 border:1px solid rgba(201,162,39,0.25);">COMING SOON</span>
  </div>
  <p style="color:{COLORS['neutral']};font-size:0.78rem;margin:0 0 6px;">{desc}</p>
  <ul style="margin:0;padding-left:16px;">{ex_html}</ul>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INGEST & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

with tab_ingest:

    _fred_ok = bool(FRED_API_KEY)
    _modes   = (
        ["FRED — Federal Reserve", "Upload File (CSV / JSON)"]
        if _fred_ok else
        ["Upload File (CSV / JSON)", "FRED (add FRED_API_KEY to .env to enable)"]
    )
    _mode    = st.radio("Data source", _modes, horizontal=True, key="ic_mode")
    _is_fred = _mode.startswith("FRED") and _fred_ok
    st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FRED BRANCH
    # ══════════════════════════════════════════════════════════════════════════
    if _is_fred:
        _fred = FREDAdapter(FRED_API_KEY)
        st.markdown(section_label("Select FRED Series"), unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            _pop = st.selectbox(
                "Popular series",
                ["— pick or search below —"] + list(POPULAR_SERIES.keys()),
                format_func=lambda k: k if k.startswith("—")
                            else f"{k} — {POPULAR_SERIES.get(k, '')}",
                key="ic_pop",
            )
        with fc2:
            _cust = st.text_input("Or enter any FRED series ID",
                                   placeholder="e.g. MORTGAGE30US", key="ic_cust")

        _sid = _cust.strip().upper() if _cust.strip() \
               else (_pop if not _pop.startswith("—") else None)

        _sq = st.text_input("Search FRED catalogue",
                             placeholder="inflation, jobs…", key="ic_sq")
        if _sq:
            with st.spinner("Searching…"):
                _hits = _fred.search_series(_sq, limit=10)
            if _hits:
                st.dataframe(pd.DataFrame(_hits), use_container_width=True, hide_index=True)
                st.caption("Copy a series ID into the field above.")
            else:
                st.info("No results.")

        if _sid is None:
            st.info("Pick a series from the dropdown or enter a custom ID above.")
        else:
            with st.spinner(f"Fetching metadata for {_sid}…"):
                _info = _fred.get_series_info(_sid)

            st.markdown(
                f'<div style="background:{COLORS["card_bg"]};'
                f'border:1px solid {COLORS["border"]};'
                f'border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.80rem;">'
                f'<b style="color:{COLORS["gold"]};">{_info["id"]}</b> — '
                f'<span style="color:{COLORS["text"]};">{_info["title"]}</span><br>'
                f'<span style="color:{COLORS["text_muted"]};">'
                f'Units: {_info["units"]} &middot; Frequency: {_info["frequency"]} '
                f'&middot; Last updated: {_info["last_updated"]}</span></div>',
                unsafe_allow_html=True,
            )

            fd1, fd2 = st.columns(2)
            with fd1:
                _fs = st.date_input("Start date", value=date(2015, 1, 1), key="ic_fs")
            with fd2:
                _fe = st.date_input("End date", value=date.today(), key="ic_fe")

            _sname = st.text_input("Save as", value=f"fred_{_sid.lower()}", key="ic_sname")

            if st.button("Pull from FRED", key="ic_pull"):
                _ok, _raw = False, None
                with st.spinner(f"Pulling {_sid}…"):
                    try:
                        _raw = _fred.get_series(_sid, start=str(_fs), end=str(_fe))
                        _ok  = not _raw.is_empty()
                        if not _ok:
                            st.error("No data returned — check the series ID and date range.")
                    except Exception as _exc:
                        st.error(f"FRED API error: {_exc}")
                if _ok and _raw is not None:
                    _rpd = _raw.to_pandas()
                    st.success(f"Pulled {len(_rpd):,} rows for {_sid}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rows", f"{len(_rpd):,}")
                    m2.metric("Nulls", f"{_rpd[_sid].isnull().sum():,}")
                    m3.metric("Range",
                              f"{_rpd['date'].min().date()} to {_rpd['date'].max().date()}")
                    st.dataframe(_rpd.head(10), use_container_width=True, hide_index=True)
                    _fig = go.Figure(go.Scatter(
                        x=_rpd["date"], y=_rpd[_sid], mode="lines",
                        line=dict(color=COLORS["gold"], width=1.8), name=_sid,
                    ))
                    apply_theme(_fig, title=f"{_info['title']} ({_info['units']})", height=240)
                    st.plotly_chart(_fig, use_container_width=True, config=PLOTLY_CONFIG)
                    _raw.write_parquet(ALT_DIR / f"{_sname}.parquet")
                    _save_meta(_sname, {
                        "source": "FRED", "series_id": _sid,
                        "title": _info.get("title", ""), "units": _info.get("units", ""),
                        "frequency": _info.get("frequency", ""),
                        "last_refreshed": datetime.now().isoformat(timespec="seconds"),
                        "rows": len(_rpd), "date_from": str(_fs), "date_to": str(_fe),
                    })
                    st.success(f"Saved to data/alt/{_sname}.parquet")
                    st.info("Go to Tradability Check to analyse this signal.")

    # ══════════════════════════════════════════════════════════════════════════
    # UPLOAD BRANCH
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown(section_label("Upload Alternative Data"), unsafe_allow_html=True)
        st.caption("CSV or JSON with a date column and at least one numeric signal column.")
        _upl = st.file_uploader("Drop a file", type=["csv", "json"], key="ic_upl")

        if _upl is None:
            _saved = sorted(ALT_DIR.glob("*.parquet"))
            _meta  = _load_meta()
            if _saved:
                st.markdown(section_label("Saved Alt Data — History"), unsafe_allow_html=True)
                for _f in _saved:
                    _m  = _meta.get(_f.stem, {})
                    _sz = _f.stat().st_size / 1024
                    _ref = _m.get("last_refreshed", "—")[:19].replace("T", " ")
                    _src = {
                        "FRED":   f"FRED: {_m.get('series_id', '')}",
                        "merge":  f"Merge: {', '.join(_m.get('components', []))}",
                        "upload": f"Upload: {_m.get('filename', '')}",
                    }.get(_m.get("source", ""), "—")
                    _dr = f"{_m['date_from']} to {_m['date_to']}" if "date_from" in _m else "—"
                    _rc = f"{_m['rows']:,} rows" if "rows" in _m else ""
                    _a, _b, _c, _d, _e = st.columns([3, 2, 2, 1, 1])
                    _a.markdown(
                        f'<span style="color:{COLORS["text"]};font-size:0.83rem;'
                        f'font-weight:600;">{_f.stem}</span><br>'
                        f'<span style="color:{COLORS["text_muted"]};font-size:0.72rem;">'
                        f'{_src}</span>',
                        unsafe_allow_html=True,
                    )
                    _b.markdown(
                        f'<span style="color:{COLORS["neutral"]};font-size:0.75rem;">'
                        f'{_dr}</span>',
                        unsafe_allow_html=True,
                    )
                    _c.markdown(
                        f'<span style="color:{COLORS["text_muted"]};font-size:0.75rem;">'
                        f'{_rc} &middot; {_sz:.1f} KB<br>Last: {_ref}</span>',
                        unsafe_allow_html=True,
                    )
                    # FRED auto-refresh
                    if _m.get("source") == "FRED" and _m.get("series_id"):
                        if _e.button("Refresh", key=f"ref_{_f.stem}",
                                     help="Re-pull this series up to today"):
                            try:
                                _rfr = FREDAdapter(FRED_API_KEY)
                                _new = _rfr.get_series(
                                    _m["series_id"],
                                    start=_m.get("date_from", "2015-01-01"),
                                    end=str(date.today()),
                                )
                                if not _new.is_empty():
                                    _new.write_parquet(_f)
                                    _save_meta(_f.stem, {
                                        **_m,
                                        "last_refreshed": datetime.now().isoformat(timespec="seconds"),
                                        "rows": len(_new),
                                        "date_to": str(date.today()),
                                    })
                                    st.success(f"Refreshed {_f.stem} ({len(_new):,} rows)")
                                    st.rerun()
                                else:
                                    st.warning("No new data returned.")
                            except Exception as _rex:
                                st.error(f"Refresh failed: {_rex}")
                    else:
                        _e.write("")  # spacer for non-FRED files
                    if _d.button("Delete", key=f"del_{_f.stem}"):
                        _f.unlink()
                        if _f.stem in _meta:
                            del _meta[_f.stem]
                            _META_FILE.write_text(json.dumps(_meta, indent=2))
                        st.rerun()
            else:
                st.info("No saved alt data yet. Upload a file or pull from FRED.")
        else:
            _raw_df = None
            try:
                _txt = _upl.read().decode("utf-8")
                _raw_df = pd.read_json(StringIO(_txt)) if _upl.name.endswith(".json") \
                          else pd.read_csv(StringIO(_txt))
            except Exception as _exc:
                st.error(f"Could not parse file: {_exc}")

            if _raw_df is not None:
                st.markdown(section_label(f"Raw Preview — {_upl.name}"),
                            unsafe_allow_html=True)
                st.caption(f"{len(_raw_df):,} rows &middot; {len(_raw_df.columns)} columns")
                st.dataframe(_raw_df.head(10), use_container_width=True, hide_index=True)

                st.markdown(section_label("Column Mapping"), unsafe_allow_html=True)
                _ac = list(_raw_df.columns)
                _sc = [c for c in _ac if _raw_df[c].dtype == object]
                _nc = [c for c in _ac if pd.api.types.is_numeric_dtype(_raw_df[c])]
                _dg = next((c for c in _ac if "date" in c.lower() or "time" in c.lower()), _ac[0])
                _sg = next((c for c in _sc if any(k in c.lower()
                            for k in ["ticker","symbol","sym"])), "none")

                _mc1, _mc2, _mc3 = st.columns(3)
                with _mc1:
                    _dcol = st.selectbox("Date column", _ac,
                                         index=_ac.index(_dg), key="ic_dcol")
                with _mc2:
                    _so   = ["none"] + _sc
                    _scol = st.selectbox("Symbol column (panel data)",
                                         _so,
                                         index=_so.index(_sg) if _sg in _so else 0,
                                         key="ic_scol")
                with _mc3:
                    _sigc = st.multiselect("Signal column(s)", _nc,
                                           default=_nc[:1] if _nc else [], key="ic_sigc")

                _srcn = st.text_input(
                    "Source name",
                    value=_upl.name.rsplit(".", 1)[0].replace(" ", "_").lower(),
                    key="ic_srcn",
                )

                if not _sigc:
                    st.warning("Select at least one signal column to continue.")
                else:
                    st.markdown(section_label("Cleaning Options"), unsafe_allow_html=True)
                    _cc1, _cc2, _cc3, _cc4 = st.columns(4)
                    with _cc1:
                        _fill = st.selectbox("Fill nulls",
                            ["forward-fill", "backfill", "interpolate", "none"],
                            key="ic_fill")
                    with _cc2:
                        _clip = st.selectbox("Outlier handling",
                            ["winsorize (1-99%)", "z-score clip (+/-3)", "none"],
                            key="ic_clip")
                    with _cc3:
                        _freq = st.selectbox("Resample",
                            ["none", "daily (B)", "weekly (W-FRI)", "monthly (MS)"],
                            key="ic_freq")
                    with _cc4:
                        _norm = st.selectbox("Normalise",
                            ["none", "z-score", "min-max [0,1]", "log", "pct_change"],
                            key="ic_norm",
                            help="z-score makes signals comparable in Tradability Check.")

                    _cdf = _raw_df.copy()
                    _cdf[_dcol] = pd.to_datetime(_cdf[_dcol], errors="coerce")
                    _cdf = _cdf.dropna(subset=[_dcol]).sort_values(_dcol)
                    _keep = [_dcol] + ([_scol] if _scol != "none" else []) + _sigc
                    _cdf = _cdf[_keep].copy()

                    if   _fill == "forward-fill": _cdf[_sigc] = _cdf[_sigc].ffill()
                    elif _fill == "backfill":     _cdf[_sigc] = _cdf[_sigc].bfill()
                    elif _fill == "interpolate":  _cdf[_sigc] = _cdf[_sigc].interpolate(
                                                                    limit_direction="both")

                    if _clip == "winsorize (1-99%)":
                        for c in _sigc:
                            _cdf[c] = _cdf[c].clip(_cdf[c].quantile(0.01),
                                                    _cdf[c].quantile(0.99))
                    elif _clip == "z-score clip (+/-3)":
                        for c in _sigc:
                            _mu, _sd = _cdf[c].mean(), _cdf[c].std()
                            if _sd > 0: _cdf[c] = _cdf[c].clip(_mu - 3*_sd, _mu + 3*_sd)

                    if _freq != "none":
                        _rule = {"daily (B)": "B", "weekly (W-FRI)": "W-FRI",
                                 "monthly (MS)": "MS"}[_freq]
                        _cdf = _cdf.set_index(_dcol)
                        if _scol != "none":
                            _cdf = _cdf.groupby(_scol).resample(_rule)[_sigc].last().reset_index()
                        else:
                            _cdf = _cdf.resample(_rule)[_sigc].last().reset_index()

                    if _norm == "z-score":
                        for c in _sigc:
                            _mu, _sd = _cdf[c].mean(), _cdf[c].std()
                            if _sd > 0: _cdf[c] = (_cdf[c] - _mu) / _sd
                    elif _norm == "min-max [0,1]":
                        for c in _sigc:
                            _lo, _hi = _cdf[c].min(), _cdf[c].max()
                            if _hi > _lo: _cdf[c] = (_cdf[c] - _lo) / (_hi - _lo)
                    elif _norm == "log":
                        for c in _sigc: _cdf[c] = np.log(_cdf[c].clip(lower=1e-10))
                    elif _norm == "pct_change":
                        for c in _sigc: _cdf[c] = _cdf[c].pct_change()

                    st.markdown(section_label("Cleaned Preview"), unsafe_allow_html=True)
                    _s1, _s2, _s3 = st.columns(3)
                    _s1.metric("Rows (raw)", f"{len(_raw_df):,}")
                    _s2.metric("Rows (clean)", f"{len(_cdf):,}")
                    _s3.metric("Nulls", f"{_cdf[_sigc].isnull().sum().sum():,}")
                    st.dataframe(_cdf.head(20), use_container_width=True, hide_index=True)

                    if len(_cdf) > 1 and _dcol in _cdf.columns:
                        _fig2 = go.Figure()
                        for c in _sigc:
                            _fig2.add_trace(go.Scatter(
                                x=_cdf[_dcol], y=_cdf[c], name=c,
                                mode="lines", line=dict(width=2),
                            ))
                        apply_theme(_fig2, title="Cleaned Signal", height=240)
                        st.plotly_chart(_fig2, use_container_width=True, config=PLOTLY_CONFIG)

                    if st.button("Save", key="ic_save"):
                        _out = ALT_DIR / f"{_srcn}.parquet"
                        pl.from_pandas(_cdf.rename(columns={_dcol: "date"})).write_parquet(_out)
                        _save_meta(_srcn, {
                            "source": "upload", "filename": _upl.name,
                            "last_refreshed": datetime.now().isoformat(timespec="seconds"),
                            "rows": len(_cdf),
                            "date_from": str(_cdf[_dcol].min())[:10] if _dcol in _cdf else "—",
                            "date_to":   str(_cdf[_dcol].max())[:10] if _dcol in _cdf else "—",
                            "normalise": _norm,
                        })
                        st.success(f"Saved to data/alt/{_srcn}.parquet ({len(_cdf):,} rows)")
                        st.info("Go to Tradability Check to analyse this signal.")

    # ── Merge (always visible when 2+ files exist) ────────────────────────────
    _all_saved = sorted(ALT_DIR.glob("*.parquet"))
    if len(_all_saved) >= 2:
        st.markdown(section_label("Merge Signals"), unsafe_allow_html=True)
        st.caption("Join two or more saved signals by date into a composite dataset.")
        _mg1, _mg2, _mg3 = st.columns([3, 1.5, 1.5])
        with _mg1:
            _mf = st.multiselect("Signals to merge",
                                  [f.stem for f in _all_saved],
                                  default=[f.stem for f in _all_saved[:2]],
                                  key="ic_mf")
        with _mg2:
            _mh = st.selectbox("Join", ["inner", "outer"], key="ic_mh")
        with _mg3:
            _mfill = st.selectbox("Fill (outer)", ["forward-fill", "none"],
                                   key="ic_mfill", disabled=(_mh == "inner"))
        _mn = st.text_input("Save merged as",
                             value="composite_" + "_".join(_mf[:3]) if _mf else "composite",
                             key="ic_mn")
        if len(_mf) >= 2 and st.button("Merge & Save", key="ic_mb_btn"):
            try:
                _frames = []
                for _stem in _mf:
                    _dm = pl.read_parquet(ALT_DIR / f"{_stem}.parquet").to_pandas()
                    _dm["date"] = pd.to_datetime(_dm["date"])
                    _frames.append(_dm.set_index("date"))
                _merged = _frames[0]
                for _fr in _frames[1:]:
                    _merged = _merged.join(_fr, how=_mh)
                if _mh == "outer" and _mfill == "forward-fill":
                    _merged = _merged.ffill()
                _merged = _merged.reset_index().rename(columns={"index": "date"})
                pl.from_pandas(_merged).write_parquet(ALT_DIR / f"{_mn}.parquet")
                _save_meta(_mn, {
                    "source": "merge", "components": _mf,
                    "last_refreshed": datetime.now().isoformat(timespec="seconds"),
                    "rows": len(_merged),
                    "date_from": str(_merged["date"].min())[:10],
                    "date_to":   str(_merged["date"].max())[:10],
                })
                st.success(f"Merged to {_mn}.parquet ({len(_merged):,} rows)")
                st.dataframe(_merged.head(5), use_container_width=True, hide_index=True)
            except Exception as _exc:
                st.error(f"Merge failed: {_exc}")

    # ── Scheduled FRED pulls ───────────────────────────────────────────────────
    _SCHED_FILE = ALT_DIR / ".fred_schedule.json"
    def _load_sched() -> list:
        if _SCHED_FILE.exists():
            try: return json.loads(_SCHED_FILE.read_text(encoding="utf-8"))
            except Exception: pass
        return []

    st.markdown(section_label("Scheduled FRED Pulls"), unsafe_allow_html=True)
    st.caption(
        "Series listed here are automatically refreshed when the daily pipeline runs. "
        "Add a series ID and it will be pulled and saved to data/alt/ each weekday."
    )
    _sched = _load_sched()
    _se1, _se2 = st.columns([3, 1])
    with _se1:
        _new_sid = st.text_input("Add FRED series ID to schedule",
                                  placeholder="UNRATE, T10Y2Y, VIXCLS…",
                                  key="sched_add")
    with _se2:
        st.markdown("<div style='height:28px'/>", unsafe_allow_html=True)
        if st.button("Add to Schedule", key="sched_btn") and _new_sid.strip():
            _sid_up = _new_sid.strip().upper()
            if _sid_up not in _sched:
                _sched.append(_sid_up)
                _SCHED_FILE.write_text(json.dumps(_sched, indent=2), encoding="utf-8")
                st.success(f"{_sid_up} added to schedule")
                st.rerun()
    if _sched:
        for _ss in list(_sched):
            _sa, _sb = st.columns([5, 1])
            _sa.markdown(
                f'<span style="color:{COLORS["gold"]};font-family:monospace;">{_ss}</span>',
                unsafe_allow_html=True,
            )
            if _sb.button("Remove", key=f"rm_{_ss}"):
                _sched.remove(_ss)
                _SCHED_FILE.write_text(json.dumps(_sched, indent=2), encoding="utf-8")
                st.rerun()
    else:
        st.caption("No series scheduled yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADABILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

with tab_trade:

    _tfiles = sorted(ALT_DIR.glob("*.parquet"))
    if not _tfiles:
        st.info("No saved alt data yet. Pull from FRED or upload a file in Ingest & Clean.")
    else:
        # ── Multi-signal comparison ────────────────────────────────────────────
        with st.expander("Compare All Signals", expanded=False):
            st.caption("Ranks every saved signal by best IC against the selected equity target.")
            _cmp_eq = st.selectbox("Equity target for comparison",
                                    ["SPY", "QQQ", "IWM"], key="cmp_eq")
            if st.button("Run Comparison", key="cmp_run"):
                _rows = []
                _cmp_bars = load_bars([_cmp_eq], date(2015,1,1), date.today(), "equity")
                _cmp_pc   = "adj_close" if "adj_close" in _cmp_bars.columns else "close"
                _cmp_px   = _cmp_bars.to_pandas().set_index("date")[_cmp_pc]
                _cmp_px.index = pd.to_datetime(_cmp_px.index)
                for _cf in _tfiles:
                    try:
                        _cpd = pl.read_parquet(_cf).to_pandas()
                        if "date" in _cpd.columns:
                            _cpd["date"] = pd.to_datetime(_cpd["date"])
                            _cpd = _cpd.set_index("date")
                        _cnc = [c for c in _cpd.columns if pd.api.types.is_numeric_dtype(_cpd[c])]
                        for _cc in _cnc[:1]:
                            _cs = _cpd[_cc].dropna()
                            _best_ic, _best_lag = 0.0, "—"
                            for _cl in [1, 5, 21, 63]:
                                _cf2 = _cmp_px.pct_change(_cl).shift(-_cl)
                                _ca  = pd.concat([_cs.rename("s"), _cf2.rename("r")], axis=1).dropna()
                                if len(_ca) >= 10:
                                    _ci, _ = spearmanr(_ca["s"], _ca["r"])
                                    if abs(float(_ci)) > abs(_best_ic):
                                        _best_ic, _best_lag = float(_ci), f"{_cl}d"
                            _rows.append({"signal": _cf.stem, "column": _cc,
                                          "best_IC": round(_best_ic, 4), "best_lag": _best_lag})
                    except Exception:
                        pass
                if _rows:
                    _cdf2 = pd.DataFrame(_rows).sort_values("best_IC", key=abs, ascending=False)
                    st.dataframe(_cdf2.style.format({"best_IC": "{:+.4f}"}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.warning("Could not compute IC for any saved signal.")

        _tc1, _tc2, _tc3, _tc4 = st.columns([2, 1.5, 1.5, 1])
        with _tc1:
            _tsrc = st.selectbox("Alt data source", [f.stem for f in _tfiles], key="tc_src")
        with _tc2:
            _teq  = st.selectbox("Equity target",
                                  ["SPY", "QQQ", "IWM", "Equal-weight universe"],
                                  key="tc_eq")
        with _tc4:
            st.markdown("<div style='height:28px'/>", unsafe_allow_html=True)
            _trun = st.button("Run Analysis", key="tc_run", use_container_width=True)

        _tpl, _tpd, _tnum = None, None, []
        try:
            _tpl  = pl.read_parquet(ALT_DIR / f"{_tsrc}.parquet")
            _tpd  = _tpl.to_pandas()
            if "date" in _tpd.columns:
                _tpd["date"] = pd.to_datetime(_tpd["date"])
                _tpd = _tpd.sort_values("date").set_index("date")
            _tnum = [c for c in _tpd.columns if pd.api.types.is_numeric_dtype(_tpd[c])]
        except Exception as _exc:
            st.error(f"Could not load {_tsrc}: {_exc}")

        if _tnum and _tpd is not None:
            with _tc3:
                _tsig = st.selectbox("Signal column", _tnum, key="tc_sig")

            if not _trun:
                st.markdown(section_label("Source Preview"), unsafe_allow_html=True)
                st.dataframe(_tpd.reset_index().head(10),
                             use_container_width=True, hide_index=True)
                st.caption(f"{len(_tpd):,} rows")
            else:
                _tstart = _tpd.index.min().date() if hasattr(_tpd.index.min(), "date") \
                          else date(2019, 1, 1)
                _tend   = _tpd.index.max().date() if hasattr(_tpd.index.max(), "date") \
                          else date.today()

                with st.spinner("Loading equity prices…"):
                    if _teq == "Equal-weight universe":
                        from storage.universe import universe_as_of_date
                        _syms = universe_as_of_date("equity", _tend)[:10]
                        _bars = load_bars(_syms, _tstart, _tend, "equity")
                        _pc   = "adj_close" if "adj_close" in _bars.columns else "close"
                        _eqpx = (_bars.to_pandas()
                                      .pivot(index="date", columns="symbol", values=_pc)
                                      .mean(axis=1))
                    else:
                        _bars = load_bars([_teq], _tstart, _tend, "equity")
                        _pc   = "adj_close" if "adj_close" in _bars.columns else "close"
                        _eqpx = _bars.to_pandas().set_index("date")[_pc]

                _eqpx.index = pd.to_datetime(_eqpx.index)
                _sig = _tpd[_tsig].dropna()

                _lags   = [1, 5, 21, 63]
                _lagres = []
                for _lag in _lags:
                    _fwd = _eqpx.pct_change(_lag).shift(-_lag)
                    _aln = pd.concat([_sig.rename("s"), _fwd.rename("r")], axis=1).dropna()
                    if len(_aln) >= 10:
                        _ic, _pv = spearmanr(_aln["s"], _aln["r"])
                        _lagres.append({"lag": f"{_lag}d", "IC": round(float(_ic), 4),
                                        "p_value": round(float(_pv), 4),
                                        "n_obs": len(_aln)})
                    else:
                        _lagres.append({"lag": f"{_lag}d", "IC": float("nan"),
                                        "p_value": float("nan"), "n_obs": len(_aln)})
                _ldf = pd.DataFrame(_lagres)

                _valid = _ldf.dropna(subset=["IC"])
                if not _valid.empty:
                    _best    = _valid.loc[_valid["IC"].abs().idxmax()]
                    _bic     = float(_best["IC"])
                    _bpv     = float(_best["p_value"])
                    _verdict = ("Strong signal" if abs(_bic) > 0.05 and _bpv < 0.05
                                else "Weak signal" if abs(_bic) > 0.02 else "No signal")
                    _vc = (COLORS["positive"] if "Strong" in _verdict
                           else COLORS["warning"] if "Weak" in _verdict
                           else COLORS["negative"])

                    st.markdown(section_label("Tradability Summary"), unsafe_allow_html=True)
                    _k1, _k2, _k3, _k4 = st.columns(4)
                    _k1.markdown(kpi_card("Best IC", f"{_bic:+.4f}", accent=_vc),
                                 unsafe_allow_html=True)
                    _k2.markdown(kpi_card("Best Lag", _best["lag"], accent=COLORS["gold"]),
                                 unsafe_allow_html=True)
                    _k3.markdown(kpi_card("p-value", f"{_bpv:.4f}",
                                          accent=COLORS["positive"] if _bpv < 0.05
                                          else COLORS["negative"]),
                                 unsafe_allow_html=True)
                    _k4.markdown(kpi_card("Verdict", _verdict, accent=_vc),
                                 unsafe_allow_html=True)
                    st.markdown(
                        status_banner(
                            f"{_verdict} — IC {_bic:+.4f} at {_best['lag']} forward return "
                            f"(p={_bpv:.4f}, n={int(_best['n_obs'])})",
                            color=_vc,
                        ),
                        unsafe_allow_html=True,
                    )

                    # Promote to feature pipeline
                    _ALT_FEAT = DATA_DIR / "alt" / ".promoted.json"
                    if st.button("Promote to Feature Pipeline", key="tc_promote",
                                 help="Write this signal to config so it can be used as a feature"):
                        try:
                            _prom = json.loads(_ALT_FEAT.read_text()) if _ALT_FEAT.exists() else {}
                            _prom[_tsrc] = {
                                "signal_column": _tsig,
                                "equity_target": _teq,
                                "best_IC":       _bic,
                                "best_lag":      str(_best["lag"]),
                                "verdict":       _verdict,
                                "promoted_at":   datetime.now().isoformat(timespec="seconds"),
                                "source_file":   f"data/alt/{_tsrc}.parquet",
                            }
                            _ALT_FEAT.write_text(json.dumps(_prom, indent=2))
                            st.success(
                                f"Promoted! Entry written to `data/alt/.promoted.json`. "
                                f"Reference this in `features/canonical.py` to include "
                                f"`{_tsrc}/{_tsig}` as a formal feature."
                            )
                        except Exception as _pe:
                            st.error(f"Promote failed: {_pe}")

                st.markdown(section_label("IC at Multiple Forward Return Horizons"),
                            unsafe_allow_html=True)
                _bc = [COLORS["positive"] if v > 0.05
                       else COLORS["negative"] if v < -0.05
                       else COLORS["gold_dim"]
                       for v in _ldf["IC"].fillna(0)]
                _fig_l = go.Figure(go.Bar(
                    x=_ldf["lag"], y=_ldf["IC"],
                    marker=dict(color=_bc, line=dict(width=0)),
                    text=[f"{v:+.4f}" if not pd.isna(v) else "n/a" for v in _ldf["IC"]],
                    textposition="outside",
                ))
                _fig_l.add_hline(y=0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"),
                                 annotation_text="+0.05")
                _fig_l.add_hline(y=-0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
                _fig_l.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1))
                apply_theme(_fig_l, title="Spearman IC by Forward Return Horizon", height=280)
                _fig_l.update_layout(showlegend=False)
                st.plotly_chart(_fig_l, use_container_width=True, config=PLOTLY_CONFIG)
                st.dataframe(_ldf.style.format({"IC": "{:+.4f}", "p_value": "{:.4f}"}),
                             use_container_width=True, hide_index=True)

                _fwd21  = _eqpx.pct_change(21).shift(-21)
                _aln21  = pd.concat([_sig.rename("s"), _fwd21.rename("r")], axis=1).dropna()
                _win    = max(21, len(_aln21) // 10)
                _rollrc = []
                for _i in range(_win, len(_aln21)):
                    _ch = _aln21.iloc[_i-_win:_i]
                    _ic_r, _ = spearmanr(_ch["s"], _ch["r"])
                    _rollrc.append({"date": _aln21.index[_i], "rolling_ic": float(_ic_r)})

                if _rollrc:
                    _rdf = pd.DataFrame(_rollrc)
                    st.markdown(section_label(f"Rolling IC — {_win}-period window"),
                                unsafe_allow_html=True)
                    _fig_r = go.Figure()
                    _fig_r.add_hline(y=0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
                    _fig_r.add_hline(y=-0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
                    _fig_r.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1))
                    _fig_r.add_trace(go.Scatter(x=_rdf["date"], y=_rdf["rolling_ic"],
                                                mode="lines",
                                                line=dict(color=COLORS["gold"], width=1.8),
                                                name="Rolling IC"))
                    _fig_r.add_trace(go.Scatter(
                        x=_rdf["date"],
                        y=_rdf["rolling_ic"].rolling(5, min_periods=1).mean(),
                        mode="lines",
                        line=dict(color=COLORS["green"], width=1, dash="dot"),
                        name="5-period MA",
                    ))
                    apply_theme(_fig_r, title="Rolling IC", height=260, legend_inside=True)
                    st.plotly_chart(_fig_r, use_container_width=True, config=PLOTLY_CONFIG)

                if not _valid.empty:
                    _bl  = int(str(_best["lag"]).replace("d", ""))
                    _scd = pd.concat(
                        [_sig.rename("s"), _eqpx.pct_change(_bl).shift(-_bl).rename("r")],
                        axis=1,
                    ).dropna()
                    if len(_scd) > 5:
                        _m, _b = np.polyfit(_scd["s"], _scd["r"], 1)
                        _xl    = np.linspace(_scd["s"].min(), _scd["s"].max(), 50)
                        st.markdown(
                            section_label(f"Signal vs {_best['lag']} Forward Return — Scatter"),
                            unsafe_allow_html=True,
                        )
                        _fig_s = go.Figure()
                        _fig_s.add_trace(go.Scatter(
                            x=_scd["s"], y=_scd["r"], mode="markers",
                            marker=dict(color=COLORS["gold"], size=5, opacity=0.6),
                            name="Observations",
                        ))
                        _fig_s.add_trace(go.Scatter(
                            x=_xl, y=_m*_xl+_b, mode="lines",
                            line=dict(color=COLORS["green"], width=2), name="OLS fit",
                        ))
                        apply_theme(_fig_s, title="", height=300, legend_inside=True)
                        _fig_s.update_layout(
                            xaxis_title=f"Signal ({_tsig})",
                            yaxis_title=f"{_best['lag']} forward return",
                            yaxis=dict(tickformat=".1%"),
                        )
                        st.plotly_chart(_fig_s, use_container_width=True, config=PLOTLY_CONFIG)

                # Signal lag scan — IC vs 21d fwd returns at different signal lags
                st.markdown(section_label("Signal Lag Scan"), unsafe_allow_html=True)
                st.caption(
                    "Shifts the signal by different amounts before computing IC. "
                    "A peak at -21 means the signal predicts returns 21 days ahead."
                )
                _scan_lags = [-63, -42, -21, -10, -5, -1, 0, 1, 5, 10, 21, 42, 63]
                _scan_fwd  = _eqpx.pct_change(21).shift(-21)
                _scan_res  = []
                for _sl in _scan_lags:
                    _ss = _sig.shift(_sl)
                    _sa = pd.concat([_ss.rename("s"), _scan_fwd.rename("r")], axis=1).dropna()
                    if len(_sa) >= 10:
                        _sic, _spv = spearmanr(_sa["s"], _sa["r"])
                        _scan_res.append({"signal_lag": _sl, "IC": round(float(_sic), 4),
                                          "p_value": round(float(_spv), 4)})
                if _scan_res:
                    _sldf  = pd.DataFrame(_scan_res)
                    _slc   = [COLORS["positive"] if v > 0.05
                               else COLORS["negative"] if v < -0.05
                               else COLORS["gold_dim"]
                               for v in _sldf["IC"]]
                    _fig_sl = go.Figure(go.Bar(
                        x=_sldf["signal_lag"], y=_sldf["IC"],
                        marker=dict(color=_slc, line=dict(width=0)),
                        hovertemplate="Lag %{x}d: IC=%{y:.4f}<extra></extra>",
                    ))
                    _fig_sl.add_hline(y=0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
                    _fig_sl.add_hline(y=-0.05, line=dict(color=COLORS["positive"], width=1, dash="dot"))
                    _fig_sl.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1))
                    apply_theme(_fig_sl,
                                title="IC vs 21d Forward Return at Different Signal Lags",
                                height=280)
                    _fig_sl.update_layout(
                        showlegend=False,
                        xaxis_title="Signal lag (negative = signal leads returns)",
                        yaxis_title="IC (Spearman)",
                    )
                    st.plotly_chart(_fig_sl, use_container_width=True, config=PLOTLY_CONFIG)
                    _best_scan = _sldf.loc[_sldf["IC"].abs().idxmax()]
                    st.caption(
                        f"Optimal signal lag: **{int(_best_scan['signal_lag'])}d** "
                        f"(IC = {_best_scan['IC']:+.4f}). "
                        f"Negative = signal leads returns; positive = signal lags returns."
                    )
