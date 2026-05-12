from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed" / "world_nations"

COUNTRY_YEAR_PATH = DATA_DIR / "world_nations_country_year_1930_2026.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "world_nations_data_dictionary.csv"
ENTITIES_PATH = DATA_DIR / "world_nations_entities_1930_2026.csv"
YEARLY_COUNTS_PATH = DATA_DIR / "world_nations_yearly_counts_1930_2026.csv"

YEAR_MIN = 1930
YEAR_MAX = 2026


@dataclass(frozen=True)
class EntityFix:
    iso_alpha2: str | None = None
    continent: str | None = None
    subregion: str | None = None
    football_confederation: str | None = None


ENTITY_FIXES = {
    "AND": EntityFix(continent="Europe", subregion="Southern Europe", football_confederation="UEFA"),
    "ATA": EntityFix(continent="Antarctica", subregion="Antarctica"),
    "ATF": EntityFix(continent="Antarctica", subregion="Antarctica"),
    "BES": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "BLM": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "BVT": EntityFix(continent="Antarctica", subregion="Antarctica"),
    "CUW": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "HMD": EntityFix(continent="Antarctica", subregion="Antarctica"),
    "HUN": EntityFix(continent="Europe", subregion="Eastern Europe", football_confederation="UEFA"),
    "MAF": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "MMR": EntityFix(continent="Asia", subregion="South-Eastern Asia", football_confederation="AFC"),
    "MNE": EntityFix(continent="Europe", subregion="Southern Europe", football_confederation="UEFA"),
    "NAM": EntityFix(iso_alpha2="NA"),
    "PSE": EntityFix(continent="Asia", subregion="Western Asia", football_confederation="AFC"),
    "SXM": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "TCA": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "UMI": EntityFix(continent="Oceania", subregion="Micronesia"),
    "VAT": EntityFix(continent="Europe", subregion="Southern Europe"),
    "VGB": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "VIR": EntityFix(continent="Americas", subregion="Caribbean", football_confederation="CONCACAF"),
    "ALA": EntityFix(continent="Europe", subregion="Northern Europe"),
}


TEXT_COLUMNS = {
    "entity_name",
    "iso_alpha2",
    "iso_alpha3",
    "entity_type",
    "sovereign_status",
    "continent",
    "subregion",
    "football_confederation",
    "source_statehood",
    "source_continent",
    "source_confederation",
    "notes",
    "column",
    "description",
    "definition",
    "primary_source",
}


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    for column in df.columns.intersection(TEXT_COLUMNS):
        df[column] = df[column].astype("string").str.strip()
        df.loc[df[column] == "", column] = pd.NA
    return df


def apply_entity_fixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for iso_alpha3, fix in ENTITY_FIXES.items():
        mask = df["iso_alpha3"].eq(iso_alpha3)
        if not mask.any():
            continue
        for column in ("iso_alpha2", "continent", "subregion", "football_confederation"):
            value = getattr(fix, column)
            if value is None:
                continue
            missing_or_unknown = df[column].isna()
            if column == "football_confederation":
                missing_or_unknown |= df[column].eq("Unknown")
            df.loc[mask & missing_or_unknown, column] = value
    return df


def clean_entities(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_entity_fixes(df)
    df["start_year"] = df["start_year"].astype("int64")
    df["end_year"] = df["end_year"].astype("int64")
    return df.sort_values(["start_year", "entity_name", "iso_alpha3"]).reset_index(drop=True)


def clean_country_year(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_entity_fixes(df)
    df["year"] = df["year"].astype("int64")
    df["official_state_count_member"] = df["official_state_count_member"].astype("int64")
    return df.sort_values(["year", "entity_name", "iso_alpha3"]).reset_index(drop=True)


def rebuild_yearly_counts(country_year: pd.DataFrame, source_counts: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        country_year.groupby("year", as_index=False)
        .agg(
            official_state_count=("official_state_count_member", "sum"),
            entities_in_panel_count=("entity_name", "size"),
            fifa_confederation_mapped_entities_count=(
                "football_confederation",
                lambda series: series.ne("Unknown").sum(),
            ),
        )
        .astype(
            {
                "year": "int64",
                "official_state_count": "int64",
                "entities_in_panel_count": "int64",
                "fifa_confederation_mapped_entities_count": "int64",
            }
        )
    )
    official = country_year[country_year["official_state_count_member"].eq(1)]
    for confederation in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA", "Unknown"):
        counts = (
            official[official["football_confederation"].eq(confederation)]
            .groupby("year")
            .size()
            .rename(f"official_state_count_{confederation}")
        )
        grouped = grouped.merge(counts, on="year", how="left")
        grouped[f"official_state_count_{confederation}"] = (
            grouped[f"official_state_count_{confederation}"].fillna(0).astype("int64")
        )

    metadata = source_counts[["year", "definition", "primary_source"]].copy()
    return grouped.merge(metadata, on="year", how="left").sort_values("year").reset_index(drop=True)


def validate(entities: pd.DataFrame, country_year: pd.DataFrame, yearly_counts: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    if country_year["year"].min() != YEAR_MIN or country_year["year"].max() != YEAR_MAX:
        issues.append("country_year does not cover 1930-2026")
    if yearly_counts["year"].min() != YEAR_MIN or yearly_counts["year"].max() != YEAR_MAX:
        issues.append("yearly_counts does not cover 1930-2026")
    if country_year.duplicated(["year", "iso_alpha3"]).any():
        issues.append("country_year contains duplicate year/iso_alpha3 rows")
    if country_year.duplicated(["year", "entity_name"]).any():
        issues.append("country_year contains duplicate year/entity_name rows")
    if entities.duplicated(["entity_name", "start_year", "end_year"]).any():
        issues.append("entities contains duplicate entity interval rows")

    expected_rows = int((entities["end_year"] - entities["start_year"] + 1).sum())
    if len(country_year) != expected_rows:
        issues.append(f"country_year row count {len(country_year)} does not match entity intervals {expected_rows}")

    recalculated = rebuild_yearly_counts(country_year, yearly_counts)
    count_columns = [column for column in yearly_counts.columns if column.startswith("official_")]
    count_columns += ["entities_in_panel_count", "fifa_confederation_mapped_entities_count"]
    count_columns = sorted(set(count_columns).intersection(recalculated.columns))
    merged = yearly_counts[["year", *count_columns]].merge(
        recalculated[["year", *count_columns]],
        on="year",
        suffixes=("_file", "_calc"),
    )
    for column in count_columns:
        if not merged[f"{column}_file"].equals(merged[f"{column}_calc"]):
            issues.append(f"yearly_counts column {column} does not match country_year")

    return issues


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def build(output_dir: Path) -> dict[str, object]:
    entities = clean_entities(read_csv(ENTITIES_PATH))
    country_year = clean_country_year(read_csv(COUNTRY_YEAR_PATH))
    data_dictionary = read_csv(DATA_DICTIONARY_PATH).reset_index(drop=True)
    source_yearly_counts = read_csv(YEARLY_COUNTS_PATH)
    yearly_counts = rebuild_yearly_counts(country_year, source_yearly_counts)

    validation_issues = validate(entities, country_year, yearly_counts)
    if validation_issues:
        raise ValueError("World nations validation failed: " + "; ".join(validation_issues))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "country_year": output_dir / "country_year_1930_2026.csv",
        "data_dictionary": output_dir / "data_dictionary.csv",
        "entities": output_dir / "entities_1930_2026.csv",
        "yearly_counts": output_dir / "yearly_counts_1930_2026.csv",
    }
    write_csv(country_year, outputs["country_year"])
    write_csv(data_dictionary, outputs["data_dictionary"])
    write_csv(entities, outputs["entities"])
    write_csv(yearly_counts, outputs["yearly_counts"])

    manifest = {
        "dataset": "world_nations",
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
        "rows": {
            "country_year": len(country_year),
            "data_dictionary": len(data_dictionary),
            "entities": len(entities),
            "yearly_counts": len(yearly_counts),
        },
        "outputs": {key: str(path.relative_to(ROOT)).replace("\\", "/") for key, path in outputs.items()},
        "source_files": {
            "country_year": str(COUNTRY_YEAR_PATH.relative_to(ROOT)).replace("\\", "/"),
            "data_dictionary": str(DATA_DICTIONARY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "entities": str(ENTITIES_PATH.relative_to(ROOT)).replace("\\", "/"),
            "yearly_counts": str(YEARLY_COUNTS_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "cleaning": {
            "text_columns_trimmed": True,
            "blank_text_normalized_to_empty_csv_fields": True,
            "targeted_entity_fixes": sorted(ENTITY_FIXES),
            "yearly_counts_rebuilt_from_country_year": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed world nations dataset.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build(args.output_dir)
    print(json.dumps(manifest["rows"], indent=2))


if __name__ == "__main__":
    main()
