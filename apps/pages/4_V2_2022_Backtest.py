from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.home import configure_page, render_v2_2022_backtest_dashboard


configure_page("World Cup 2022 V2 Backtest")
st.sidebar.caption("Version-isolated page")
render_v2_2022_backtest_dashboard()
