from pathlib import Path
import importlib
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import home as home
from apps import team_report_card as team_report_card


home = importlib.reload(home)
team_report_card = importlib.reload(team_report_card)
configure_page = home.configure_page
render_team_report_card_page = team_report_card.render_team_report_card_page


configure_page("World Cup 2026 Team Report Card")
st.sidebar.caption("Version-isolated page")
render_team_report_card_page()
