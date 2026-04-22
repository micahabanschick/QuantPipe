"""Live Trading Dashboard — consolidated into Paper Trading.

The live and paper trading views have been merged into a single page.
Use the Paper Trading page with the "Live" mode toggle at the top to monitor
your real-money IBKR account.
"""

import streamlit as st
from reports._theme import COLORS

st.markdown("## Live Trading — Consolidated")
st.markdown(
    f"""
<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['warning']};
     border-radius:8px;padding:20px 24px;margin-top:10px;">
  <div style="color:{COLORS['warning']};font-weight:700;font-size:1.0rem;margin-bottom:10px;">
    Live Trading has been merged into Paper Trading
  </div>
  <div style="color:{COLORS['neutral']};font-size:0.88rem;line-height:1.9;">
    Use the <strong>Paper Trading</strong> page and select <strong>Live</strong>
    in the mode toggle at the top of the page to monitor your real-money IBKR account.<br><br>
    To place live orders, go to <strong>Portfolio → Trade</strong> and select
    <em>Live Trading</em> mode.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if st.button("Go to Paper / Live Trading", type="primary"):
    st.switch_page("reports/paper_trading_dashboard.py")
