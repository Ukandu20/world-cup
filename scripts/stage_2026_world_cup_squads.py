from __future__ import annotations

import csv
import html
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WORLD_CUP_DIR = ROOT / "INT-World Cup" / "world_cup"
OUTPUT_DIR = WORLD_CUP_DIR / "2026"
ALL_EDITIONS_DIR = WORLD_CUP_DIR / "all_editions"

SOURCE_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
SOURCE_AS_OF = "2026-05-22"
TOURNAMENT_ID = "WC-2026"
TOURNAMENT_NAME = "2026 FIFA Men's World Cup"

SQUAD_FIELDS_2026 = [
    "tournament_id",
    "edition_year",
    "year",
    "team_id",
    "team",
    "team_code",
    "confederation",
    "group",
    "coach",
    "player_name",
    "position",
    "pos_code",
    "jersey_number",
    "date_of_birth",
    "club",
    "caps",
    "goals",
    "is_captain",
    "is_final_squad",
    "source_url",
    "source_as_of",
]

ALL_EDITIONS_FIELDS = [
    "team",
    "team_id",
    "team_code",
    "confederation",
    "tournament",
    "tournament_id",
    "year",
    "player_name",
    "player_id",
    "position",
    "pos_code",
    "shirt_number",
    "date_of_birth",
    "age",
    "club",
    "club_country",
    "caps",
    "goals",
    "is_captain",
    "group",
    "coach",
    "is_final_squad",
    "source_url",
    "source_as_of",
]

POSITION_LABELS = {
    "GK": "goalkeeper",
    "DF": "defender",
    "MF": "midfielder",
    "FW": "forward",
}

TEAM_NAME_ALIASES = {
    "czech republic": "czechia",
    "usa": "united states",
    "united states of america": "united states",
    "south korea": "korea republic",
    "curaçao": "curacao",
    "cote d ivoire": "ivory coast",
    "côte d ivoire": "ivory coast",
    "turkiye": "turkey",
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    parent: "Node | None" = None


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {name.lower(): value or "" for name, value in attrs}, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if tag.lower() not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].text_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.stack[-1].text_parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self.stack[-1].text_parts.append(html.unescape(f"&#{name};"))


def iter_nodes(node: Node):
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def text_content(node: Node) -> str:
    parts: list[str] = []

    def collect(current: Node) -> None:
        parts.extend(current.text_parts)
        for child in current.children:
            collect(child)

    collect(node)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def direct_children(node: Node, tags: set[str]) -> list[Node]:
    return [child for child in node.children if child.tag in tags]


def descendants(node: Node, tags: set[str] | None = None) -> list[Node]:
    found: list[Node] = []
    for child in node.children:
        if tags is None or child.tag in tags:
            found.append(child)
        found.extend(descendants(child, tags))
    return found


def normalize_key(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-zA-Z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip().lower()


def clean_heading_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("[edit]", "")).strip()


def clean_player_name(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = re.sub(r"\s*\(\)\s*captain\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(captain\)\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\bcaptain\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\[[a-z0-9;,\s]+\]\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_is_captain(value: str) -> bool:
    return bool(re.search(r"\(captain\)|\bcaptain\b|\[c\]", value, flags=re.IGNORECASE))


def infer_is_final_squad(text: str) -> bool:
    text_l = re.sub(r"\s+", " ", text.lower()).strip()
    final_patterns = [
        r"\bannounced (?:their|the|his)?\s*final squad\b",
        r"\bfinal squad was announced\b",
        r"\bofficially confirmed\b",
    ]
    non_final_patterns = [
        r"\bpreliminary squad\b",
        r"\bprovisional squad\b",
        r"\bwill announce\b",
        r"\bwill be announced\b",
        r"\bsquad was reduced\b",
    ]
    if any(re.search(pattern, text_l) for pattern in final_patterns):
        return True
    if any(re.search(pattern, text_l) for pattern in non_final_patterns):
        return False
    return False


def parse_dob(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", text)
    if not match:
        return ""
    try:
        return date.fromisoformat(str(__import__("datetime").datetime.strptime(match.group(1), "%B %d, %Y").date())).isoformat()
    except ValueError:
        return ""


def int_or_blank(value: str) -> str:
    text = re.sub(r"\D+", "", value)
    return text if text else ""


def extract_pos_code(value: str) -> str:
    match = re.search(r"\b(GK|DF|MF|FW)\b", value.upper())
    if match:
        return match.group(1)
    match = re.search(r"(GK|DF|MF|FW)", value.upper())
    return match.group(1) if match else ""


def should_keep_jersey_numbers(numbers: list[str]) -> bool:
    cleaned = [value.strip() for value in numbers if value.strip().isdigit()]
    unique = set(cleaned)
    return len(cleaned) >= 23 and len(unique) == len(cleaned) and all(1 <= int(value) <= 99 for value in cleaned)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize(row.get(field, "")) for field in fieldnames})


def serialize(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def fetch_html(url: str) -> str:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_document(raw_html: str) -> Node:
    parser = TreeBuilder()
    parser.feed(raw_html)
    parser.close()
    return parser.root


def block_nodes(root: Node) -> list[Node]:
    wanted = {"h2", "h3", "p", "table"}
    return [node for node in iter_nodes(root) if node.tag in wanted]


def cell_rows(table: Node) -> list[list[Node]]:
    rows: list[list[Node]] = []
    for tr in descendants(table, {"tr"}):
        cells = direct_children(tr, {"th", "td"})
        if cells:
            rows.append(cells)
    return rows


def table_headers(table: Node) -> list[str]:
    for row in cell_rows(table):
        headers = [text_content(cell) for cell in row]
        if {"No.", "Pos.", "Player"}.issubset(set(headers)):
            return headers
    return []


def looks_like_squad_table(table: Node) -> bool:
    headers = set(table_headers(table))
    return {"No.", "Pos.", "Player", "Date of birth (age)", "Caps", "Goals", "Club"}.issubset(headers)


def extract_club(cell: Node) -> str:
    labels: list[str] = []
    for link in descendants(cell, {"a"}):
        label = text_content(link)
        href = link.attrs.get("href", "")
        if not label:
            continue
        if href.startswith("/wiki/File:") or "Image:" in label:
            continue
        labels.append(label)
    return labels[-1] if labels else text_content(cell)


def normalize_table(table: Node, context: dict[str, object], metadata: dict[str, str]) -> list[dict[str, object]]:
    headers = table_headers(table)
    index_by_header = {name: index for index, name in enumerate(headers)}
    data_rows = []
    raw_rows = cell_rows(table)
    header_seen = False
    for cells in raw_rows:
        labels = [text_content(cell) for cell in cells]
        if labels == headers:
            header_seen = True
            continue
        if not header_seen or len(cells) < len(headers):
            continue
        data_rows.append(cells)

    numbers = [text_content(row[index_by_header["No."]]) for row in data_rows if len(row) > index_by_header["No."]]
    keep_numbers = should_keep_jersey_numbers(numbers)
    rows: list[dict[str, object]] = []

    for cells in data_rows:
        def get(header: str) -> str:
            index = index_by_header.get(header, -1)
            if index < 0 or index >= len(cells):
                return ""
            return text_content(cells[index])

        player_raw = get("Player")
        player_name = clean_player_name(player_raw)
        pos_code = extract_pos_code(get("Pos."))
        if not player_name or pos_code not in POSITION_LABELS:
            continue
        no_value = get("No.").strip()
        club_cell = cells[index_by_header["Club"]] if index_by_header.get("Club", -1) < len(cells) else Node("td")
        rows.append(
            {
                "tournament_id": TOURNAMENT_ID,
                "edition_year": "2026",
                "year": "2026",
                "team_id": metadata.get("team_id", ""),
                "team": metadata.get("team", context["team_name"]),
                "team_code": metadata.get("team_code", metadata.get("team_id", "")),
                "confederation": metadata.get("confederation", ""),
                "group": context["group"],
                "coach": context["coach"],
                "player_name": player_name,
                "position": POSITION_LABELS[pos_code],
                "pos_code": pos_code,
                "jersey_number": no_value if keep_numbers and no_value.isdigit() else "",
                "date_of_birth": parse_dob(get("Date of birth (age)")),
                "club": extract_club(club_cell),
                "caps": int_or_blank(get("Caps")),
                "goals": int_or_blank(get("Goals")),
                "is_captain": infer_is_captain(player_raw),
                "is_final_squad": context["is_final_squad"],
                "source_url": SOURCE_URL,
                "source_as_of": SOURCE_AS_OF,
            }
        )
    return rows


def load_team_metadata() -> dict[str, dict[str, str]]:
    teams = load_csv(OUTPUT_DIR / "teams.csv")
    groups = load_csv(OUTPUT_DIR / "groups.csv")
    group_by_team_id = {row["team_id"]: row["group_code"] for row in groups}
    metadata: dict[str, dict[str, str]] = {}
    for row in teams:
        values = {
            "team_id": row["team_id"],
            "team": row["team"],
            "canonical_name": row.get("canonical_name", row["team"]),
            "team_code": row.get("fifa_code", row["team_id"]),
            "confederation": row["confederation"],
            "group_code": group_by_team_id.get(row["team_id"], row.get("group_code", "")),
        }
        keys = {row["team"], row.get("canonical_name", ""), row.get("tournament_name", "")}
        for key in keys:
            if key:
                metadata[normalize_key(key)] = values
    for source, target in TEAM_NAME_ALIASES.items():
        target_values = metadata.get(normalize_key(target))
        if target_values:
            metadata[normalize_key(source)] = target_values
    return metadata


def scrape_squads() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metadata_by_name = load_team_metadata()
    root = parse_document(fetch_html(SOURCE_URL))
    active_group = ""
    active_context: dict[str, object] | None = None
    announcement_parts: list[str] = []
    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    team_status: dict[str, dict[str, object]] = {}

    for node in block_nodes(root):
        text = clean_heading_text(text_content(node))
        group_match = re.fullmatch(r"Group ([A-L])", text)
        if node.tag == "h2":
            if active_context and active_context["team_name"] not in team_status:
                metadata = metadata_by_name.get(normalize_key(str(active_context["team_name"])), {})
                team_status[active_context["team_name"]] = make_status_row(active_context, metadata, 0)
                warnings.append(warning_row(active_context, "warning_missing_table", "No squad table found for team section."))
            active_group = group_match.group(1) if group_match else ""
            active_context = None
            announcement_parts = []
            continue

        if node.tag == "h3" and active_group:
            if active_context and active_context["team_name"] not in team_status:
                metadata = metadata_by_name.get(normalize_key(str(active_context["team_name"])), {})
                team_status[active_context["team_name"]] = make_status_row(active_context, metadata, 0)
                warnings.append(warning_row(active_context, "warning_missing_table", "No squad table found for team section."))
            team_name = text
            active_context = {
                "team_name": team_name,
                "group": active_group,
                "coach": "",
                "announcement_text": "",
                "is_final_squad": False,
            }
            announcement_parts = []
            continue

        if not active_context:
            continue

        if node.tag == "p":
            paragraph = text
            if not paragraph:
                continue
            coach_match = re.search(r"\bCoach:\s*([^\.]+)", paragraph)
            if coach_match and not active_context["coach"]:
                active_context["coach"] = coach_match.group(1).strip()
            else:
                announcement_parts.append(paragraph)
            continue

        if node.tag == "table" and looks_like_squad_table(node):
            active_context["announcement_text"] = " ".join(announcement_parts).strip()
            active_context["is_final_squad"] = infer_is_final_squad(str(active_context["announcement_text"]))
            metadata = metadata_by_name.get(normalize_key(str(active_context["team_name"])), {})
            extracted = normalize_table(node, active_context, metadata)
            rows.extend(extracted)
            team_status[active_context["team_name"]] = make_status_row(active_context, metadata, len(extracted))
            warnings.extend(validate_team(active_context, metadata, extracted))
            active_context = None
            announcement_parts = []

    if active_context and active_context["team_name"] not in team_status:
        metadata = metadata_by_name.get(normalize_key(str(active_context["team_name"])), {})
        team_status[active_context["team_name"]] = make_status_row(active_context, metadata, 0)
        warnings.append(warning_row(active_context, "warning_missing_table", "No squad table found for team section."))

    duplicate_keys: set[tuple[str, str, str]] = set()
    seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["team"]), str(row["player_name"]), str(row["date_of_birth"]))
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    for team, player_name, dob in sorted(duplicate_keys):
        warnings.append(
            {
                "team_name": team,
                "group": "",
                "is_final_squad": "",
                "row_count": "",
                "expected_count": "",
                "status": "warning_duplicate_player",
                "message": f"Duplicate player row: {player_name} ({dob}).",
            }
        )

    return rows, list(team_status.values()), warnings


def make_status_row(context: dict[str, object], metadata: dict[str, str], row_count: int) -> dict[str, object]:
    return {
        "team_name": metadata.get("team", context["team_name"]),
        "team_id": metadata.get("team_id", ""),
        "team_code": metadata.get("team_code", metadata.get("team_id", "")),
        "confederation": metadata.get("confederation", ""),
        "group": context["group"],
        "coach": context["coach"],
        "is_final_squad": context["is_final_squad"],
        "row_count": row_count,
        "announcement_text": context.get("announcement_text", ""),
        "source_url": SOURCE_URL,
        "source_as_of": SOURCE_AS_OF,
    }


def warning_row(context: dict[str, object], status: str, message: str, row_count: int = 0) -> dict[str, object]:
    return {
        "team_name": context["team_name"],
        "group": context["group"],
        "is_final_squad": context.get("is_final_squad", False),
        "row_count": row_count,
        "expected_count": 26 if context.get("is_final_squad") else "",
        "status": status,
        "message": message,
    }


def validate_team(context: dict[str, object], metadata: dict[str, str], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    row_count = len(rows)
    if not metadata:
        warnings.append(warning_row(context, "warning_missing_metadata", "No local team metadata match found.", row_count))
    if not context.get("coach"):
        warnings.append(warning_row(context, "warning_missing_coach", "No coach parsed for team.", row_count))
    if context.get("is_final_squad") and row_count != 26:
        warnings.append(
            warning_row(
                context,
                "warning_final_squad_not_26",
                f"Final squad has {row_count} players; project expects 26.",
                row_count,
            )
        )
    if not context.get("is_final_squad"):
        warnings.append(warning_row(context, "warning_non_final_squad", f"Squad is not finalized; extracted {row_count} listed players.", row_count))
    required = ["player_name", "position", "date_of_birth", "club"]
    for index, row in enumerate(rows, start=1):
        missing = [field for field in required if not row.get(field)]
        if missing:
            warnings.append(
                {
                    "team_name": row.get("team", context["team_name"]),
                    "group": context["group"],
                    "is_final_squad": context.get("is_final_squad", False),
                    "row_count": row_count,
                    "expected_count": 26 if context.get("is_final_squad") else "",
                    "status": "warning_missing_required_field",
                    "message": f"Row {index} missing: {', '.join(missing)}.",
                }
            )
    return warnings


def to_all_editions_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "team": row["team"],
        "team_id": row["team_id"],
        "team_code": row["team_code"],
        "confederation": row["confederation"],
        "tournament": TOURNAMENT_NAME,
        "tournament_id": row["tournament_id"],
        "year": row["year"],
        "player_name": row["player_name"],
        "player_id": "",
        "position": row["position"],
        "pos_code": row["pos_code"],
        "shirt_number": row["jersey_number"],
        "date_of_birth": row["date_of_birth"],
        "age": "",
        "club": row["club"],
        "club_country": "",
        "caps": row["caps"],
        "goals": row["goals"],
        "is_captain": row["is_captain"],
        "group": row["group"],
        "coach": row["coach"],
        "is_final_squad": row["is_final_squad"],
        "source_url": row["source_url"],
        "source_as_of": row["source_as_of"],
    }


def replace_all_editions(rows_2026: list[dict[str, object]]) -> None:
    path = ALL_EDITIONS_DIR / "squads.csv"
    existing = load_csv(path)
    kept = [
        row
        for row in existing
        if row.get("tournament_id") != TOURNAMENT_ID and row.get("year") != "2026"
    ]
    for row in kept:
        for field in ALL_EDITIONS_FIELDS:
            row.setdefault(field, "")
    write_csv(path, kept + [to_all_editions_row(row) for row in rows_2026], ALL_EDITIONS_FIELDS)


def main() -> None:
    squad_rows, status_rows, warning_rows = scrape_squads()
    squad_rows = sorted(squad_rows, key=lambda row: (str(row["group"]), str(row["team"]), str(row["pos_code"]), str(row["player_name"])))
    status_rows = sorted(status_rows, key=lambda row: (str(row["group"]), str(row["team_name"])))
    warning_rows = sorted(warning_rows, key=lambda row: (str(row["team_name"]), str(row["status"]), str(row["message"])))

    write_csv(OUTPUT_DIR / "squads.csv", squad_rows, SQUAD_FIELDS_2026)
    write_csv(
        OUTPUT_DIR / "squads_teams_status.csv",
        status_rows,
        ["team_name", "team_id", "team_code", "confederation", "group", "coach", "is_final_squad", "row_count", "announcement_text", "source_url", "source_as_of"],
    )
    write_csv(
        OUTPUT_DIR / "squads_validation_report.csv",
        warning_rows,
        ["team_name", "group", "is_final_squad", "row_count", "expected_count", "status", "message"],
    )
    replace_all_editions(squad_rows)

    final_count = sum(1 for row in status_rows if row["is_final_squad"])
    print(f"teams={len(status_rows)}")
    print(f"player_rows={len(squad_rows)}")
    print(f"final_squads={final_count}")
    print(f"non_final_squads={len(status_rows) - final_count}")
    print(f"validation_warnings={len(warning_rows)}")


if __name__ == "__main__":
    main()
