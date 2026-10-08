"""Append-only, hash-chained prediction ledger.

The ledger is the project's forward-test record: predictions written BEFORE a
game, never edited afterwards. Integrity comes from two things:

  * git history (the ledger file is committed after every logging run), and
  * a SHA-256 hash chain: each row's hash covers its own fields plus the
    previous row's hash, so editing, deleting, or reordering any earlier row
    breaks every hash after it. `verify()` recomputes the chain.

Values are stored and hashed as the exact strings written to disk.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from src.config import project_path


LEDGER_PATH = project_path("ledger", "predictions.csv")

FIELDS = [
    "entry_id",
    "logged_at_utc",
    "kickoff_utc",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "model_id",
    "model_sha",
    "params_sha",
    "p_home_win",
    "home_qb",
    "away_qb",
    "diff_rating_off_total",
    "diff_rating_def",
    "market_p_home",
    "spread_line",
    "home_moneyline",
    "away_moneyline",
    "data_asof",
    "prev_hash",
    "row_hash",
]

GENESIS = "0" * 64


def _row_hash(prev_hash: str, row: dict[str, str], fields: list[str] = FIELDS) -> str:
    body = "|".join(
        f"{key}={row[key]}" for key in fields if key != "row_hash"
    )
    return hashlib.sha256(f"{prev_hash}|{body}".encode()).hexdigest()


def read_rows(path: Path = LEDGER_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append(
    rows: list[dict[str, str]],
    path: Path = LEDGER_PATH,
    *,
    fields: list[str] = FIELDS,
    key: tuple[str, ...] = ("model_id", "game_id"),
) -> int:
    """Append new rows, skipping rows whose `key` fields were already logged."""
    existing = read_rows(path)
    logged = {tuple(r[k] for k in key) for r in existing}

    prev_hash = existing[-1]["row_hash"] if existing else GENESIS
    next_id = len(existing) + 1

    fresh = []

    for row in rows:
        if tuple(str(row[k]) for k in key) in logged:
            continue

        row = {k: str(row.get(k, "")) for k in fields}
        row["entry_id"] = str(next_id)
        row["prev_hash"] = prev_hash
        row["row_hash"] = _row_hash(prev_hash, row, fields)

        prev_hash = row["row_hash"]
        next_id += 1
        fresh.append(row)

    if not fresh:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)

        if new_file:
            writer.writeheader()

        writer.writerows(fresh)

    return len(fresh)


def verify(path: Path = LEDGER_PATH, fields: list[str] = FIELDS) -> tuple[bool, str]:
    """Recompute the hash chain. Returns (ok, message)."""
    rows = read_rows(path)

    prev_hash = GENESIS

    for i, row in enumerate(rows, start=1):
        if row["entry_id"] != str(i):
            return False, f"entry {i}: out-of-sequence id {row['entry_id']}"

        if row["prev_hash"] != prev_hash:
            return False, f"entry {i}: chain broken (prev_hash mismatch)"

        if _row_hash(prev_hash, row, fields) != row["row_hash"]:
            return False, f"entry {i}: row was modified after logging"

        prev_hash = row["row_hash"]

    return True, f"ledger OK: {len(rows)} entries, chain intact"


if __name__ == "__main__":
    ok, message = verify()
    print(message)
    raise SystemExit(0 if ok else 1)
