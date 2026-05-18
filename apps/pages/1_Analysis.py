from pathlib import Path
import importlib
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import historical_eda
from apps import home


home = importlib.reload(home)
historical_eda = importlib.reload(historical_eda)
configure_page = home.configure_page
render_historical_eda_page = historical_eda.render_historical_eda_page


configure_page("Analysis")
st.sidebar.caption("Analysis companion page")
render_historical_eda_page()
