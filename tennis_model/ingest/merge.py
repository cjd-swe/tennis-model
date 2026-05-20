"""
Merge Sackmann match data with tennis-data.co.uk closing odds.

The two sources use different player name formats (e.g. "Djokovic N." vs
"Novak Djokovic"), so we join on:
  1. Date within ±3 days
  2. Fuzzy name match on both winner and loser using RapidFuzz token-sort ratio

Output is one Parquet file per tour in data/processed/.

Usage (CLI):
    uv run python -m tennis_model.ingest.merge
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from tennis_model.ingest.sackmann import load_sackmann
from tennis_model.ingest.tennis_data_uk import load_tennis_data_uk

log = logging.getLogger(__name__)

_PROCESSED_DIR = Path("data/processed")
_RAW_DIR = Path("data/raw")

# Name-match threshold — token_sort_ratio must exceed this to accept
_NAME_THRESHOLD = 72

# Date tolerance in days
_DATE_WINDOW_DAYS = 3


def merge(
    tours: list[str] | None = None,
    cache_sackmann: bool = True,
    cache_tduk: bool = True,
    output_dir: Path = _PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Merge Sackmann + tennis-data.co.uk for each tour.

    Returns a dict {"atp": DataFrame, "wta": DataFrame}.
    """
    if tours is None:
        tours = ["atp", "wta"]

    sack_cache = _RAW_DIR / "sackmann" if cache_sackmann else None
    tduk_cache = _RAW_DIR / "tennis_data_uk" if cache_tduk else None

    sack = load_sackmann(tours=tours, cache_dir=sack_cache)
    tduk = load_tennis_data_uk(tours=tours, cache_dir=tduk_cache)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    for tour in tours:
        log.info("Merging %s ...", tour)
        s = sack[sack["tour"] == tour].copy()
        t = tduk[tduk["tour"] == tour].copy()
        merged = _merge_tour(s, t)
        out_path = output_dir / f"{tour}_merged.parquet"
        merged.to_parquet(out_path, index=False)
        log.info(
            "%s: %d Sackmann, %d TDUK -> %d merged (%d with odds) → %s",
            tour.upper(),
            len(s),
            len(t),
            len(merged),
            merged["closing_odds_winner"].notna().sum(),
            out_path,
        )
        results[tour] = merged

    return results


def _merge_tour(sack: pd.DataFrame, tduk: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join Sackmann (authoritative for match results) onto TDUK (odds).
    """
    # Work with date-only numpy int64 for window arithmetic
    sack = sack.copy()
    tduk = tduk.copy()

    sack["_date_int"] = sack["tourney_date"].astype("int64") // 10**9  # unix seconds
    tduk["_date_int"] = tduk["match_date"].astype("int64") // 10**9

    # Columns to bring in from TDUK
    tduk_cols = [
        "match_date",
        "closing_odds_winner",
        "closing_odds_loser",
        "implied_prob_winner",
        "odds_b365_winner",
        "odds_b365_loser",
        "odds_ps_winner",
        "odds_ps_loser",
        "odds_avg_winner",
        "odds_avg_loser",
    ]
    tduk_cols = [c for c in tduk_cols if c in tduk.columns]

    tduk_sub = tduk[["_date_int", "winner_name", "loser_name"] + tduk_cols].copy()

    matched_rows: list[dict] = []

    for _, s_row in sack.iterrows():
        date_lo = s_row["_date_int"] - _DATE_WINDOW_DAYS * 86400
        date_hi = s_row["_date_int"] + _DATE_WINDOW_DAYS * 86400

        candidates = tduk_sub[
            (tduk_sub["_date_int"] >= date_lo) & (tduk_sub["_date_int"] <= date_hi)
        ]

        if candidates.empty:
            matched_rows.append({})
            continue

        best_idx, best_score = _best_match(
            s_row["winner_name"],
            s_row["loser_name"],
            candidates,
        )

        if best_score >= _NAME_THRESHOLD:
            row_data = candidates.loc[best_idx, tduk_cols].to_dict()
        else:
            row_data = {}

        matched_rows.append(row_data)

    odds_df = pd.DataFrame(matched_rows, index=sack.index)
    merged = pd.concat([sack.reset_index(drop=True), odds_df.reset_index(drop=True)], axis=1)
    merged = merged.drop(columns=["_date_int"], errors="ignore")
    return merged


def _best_match(
    s_winner: str,
    s_loser: str,
    candidates: pd.DataFrame,
) -> tuple[int, float]:
    """
    For each candidate row score the combined name similarity and return
    (index, best_combined_score).
    """
    best_idx = candidates.index[0]
    best_score = -1.0

    for idx, row in candidates.iterrows():
        w_score = fuzz.token_sort_ratio(
            _norm_name(s_winner), _norm_name(str(row["winner_name"]))
        )
        l_score = fuzz.token_sort_ratio(
            _norm_name(s_loser), _norm_name(str(row["loser_name"]))
        )
        combined = (w_score + l_score) / 2.0
        if combined > best_score:
            best_score = combined
            best_idx = idx

    return best_idx, best_score


def _norm_name(name: str) -> str:
    return name.lower().strip()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Merge Sackmann + tennis-data.co.uk")
    parser.add_argument("--tours", nargs="+", choices=["atp", "wta"], default=["atp", "wta"])
    args = parser.parse_args()

    results = merge(tours=args.tours)
    for tour, df in results.items():
        n_odds = df["closing_odds_winner"].notna().sum()
        print(
            f"{tour.upper()}: {len(df):,} matches, "
            f"{n_odds:,} with closing odds ({100*n_odds/len(df):.1f}%)"
        )
        print(
            df[["tourney_date", "winner_name", "loser_name",
                "total_games", "closing_odds_winner", "implied_prob_winner"]].tail(5)
        )
