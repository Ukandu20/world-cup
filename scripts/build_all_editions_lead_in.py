from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_cup_sim.validation_data import refresh_all_editions_lead_in


def main() -> None:
    output_path = refresh_all_editions_lead_in()
    print(json.dumps({"lead_in": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
