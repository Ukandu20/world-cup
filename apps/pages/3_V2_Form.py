from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.home import configure_page, render_v2_dashboard


configure_page("World Cup 2026 V2")
render_v2_dashboard()
