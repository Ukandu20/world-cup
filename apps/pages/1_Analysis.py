from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import historical_eda
from apps import home


configure_page = home.configure_page
render_historical_eda_page = historical_eda.render_historical_eda_page


configure_page("Analysis")
render_historical_eda_page()
