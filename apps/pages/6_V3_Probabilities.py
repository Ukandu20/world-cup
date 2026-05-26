from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import home as home


configure_page = home.configure_page
render_v3_probabilities_dashboard = home.render_v3_probabilities_dashboard


configure_page("World Cup 2026 V3 Probabilities")
render_v3_probabilities_dashboard()
