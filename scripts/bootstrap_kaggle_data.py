from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_KAGGLE_DIR = DATA_DIR / "raw" / "kaggle" / "international-football-results"
KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
REQUIRED_FILES = ("results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the ignored Kaggle raw files used by the dataset builders."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force KaggleHub to refresh the local download.",
    )
    return parser.parse_args()


def find_downloaded_file(filename: str, roots: list[Path]) -> Path:
    for root in roots:
        if not root.exists():
            continue
        direct_path = root / filename
        if direct_path.exists():
            return direct_path
        matches = sorted(root.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {filename} in the Kaggle download")


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required for raw-data bootstrap. Install requirements first."
        ) from exc

    RAW_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    download_path = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET,
            output_dir=str(RAW_KAGGLE_DIR),
            force_download=args.force,
        )
    )

    copied: list[str] = []
    for filename in REQUIRED_FILES:
        source_path = find_downloaded_file(filename, [download_path, RAW_KAGGLE_DIR])
        destination_path = DATA_DIR / filename
        shutil.copy2(source_path, destination_path)
        copied.append(str(destination_path.relative_to(ROOT)).replace("\\", "/"))

    print("Downloaded raw Kaggle files:")
    for path in copied:
        print(f"- {path}")


if __name__ == "__main__":
    main()
