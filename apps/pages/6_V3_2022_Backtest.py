from pathlib import Path
import importlib
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import home as home


home = importlib.reload(home)
configure_page = home.configure_page
render_v3_2022_backtest_dashboard = home.render_v3_2022_backtest_dashboard


configure_page("World Cup 2022 V3 Backtest")
st.sidebar.caption("Version-isolated page")
render_v3_2022_backtest_dashboard()
