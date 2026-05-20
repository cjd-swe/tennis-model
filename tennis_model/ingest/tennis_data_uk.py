"""
Download historical match + closing-odds CSVs from tennis-data.co.uk.

Columns of interest:
  - Date, Winner, Loser, Surface, Round, Best of, WRank, LRank, W/L sets, W/L games, Score
  - B365W/B365L  — Bet365 winner/loser ML odds
  - PSW/PSL      — Pinnacle (average) winner/loser ML odds
  - AvgW/AvgL    — market-average winner/loser ML odds

Usage (CLI):
    uv run python -m tennis_model.ingest.tennis_data_uk
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── URL templates ────────────────────────────────────────────────────────────
# tennis-data.co.uk serves files at predictable paths:
#   http://www.tennis-data.co.uk/{year}/{tour}{year}.csv
# For some years the pattern differs slightly; the _url_candidates list covers both.

_BASE = "http://www.tennis-data.co.uk"

_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "atp": (2001, 2024),
    "wta": (2007, 2024),
}

_TOUR_PREFIX = {"atp": "", "wta": "W"}  # ATP has no prefix, WTA uses "W"

# Odds columns that exist (vary by year; we take what's available)
_ODDS_COLS = ["B365W", "B365L", "PSW", "PSL", "AvgW", "AvgL", "MaxW", "MaxL"]

# Canonical internal column names after normalisation
_RENAME = {
    "Date": "match_date",
    "Tournament": "tourney_name",
    "Series": "tourney_series",
    "Court": "court",
    "Surface": "surface",
    "Round": "round",
    "Best of": "best_of",
    "Winner": "winner_name",
    "Loser": "loser_name",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "WPts": "winner_rank_points",
    "LPts": "loser_rank_points",
    "W1": "w_s1",
    "L1": "l_s1",
    "W2": "w_s2",
    "L2": "l_s2",
    "W3": "w_s3",
    "L3": "l_s3",
    "W4": "w_s4",
    "L4": "l_s4",
    "W5": "w_s5",
    "L5": "l_s5",
    "Wsets": "winner_sets",
    "Lsets": "loser_sets",
    "Comment": "comment",
    "B365W": "odds_b365_winner",
    "B365L": "odds_b365_loser",
    "PSW": "odds_ps_winner",
    "PSL": "odds_ps_loser",
    "AvgW": "odds_avg_winner",
    "AvgL": "odds_avg_loser",
    "MaxW": "odds_max_winner",
    "MaxL": "odds_max_loser",
}


def _url_candidates(tour: Literal["atp", "wta"], year: int) -> list[str]:
    prefix = _TOUR_PREFIX[tour]
    return [
        f"{_BASE}/{year}/{prefix}{year}.csv",
        f"{_BASE}/{year}/{prefix}{year}.xls",  # some early years
    ]


def _fetch_year(
    tour: Literal["atp", "wta"], year: int
) -> pd.DataFrame | None:
    for url in _url_candidates(tour, year):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            if url.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(resp.content))
            else:
                # tennis-data.co.uk CSVs are sometimes latin-1 encoded
                df = pd.read_csv(
                    io.StringIO(resp.content.decode("latin-1")), low_memory=False
                )
            df["tour"] = tour
            df["year"] = year
            return df
        except Exception as exc:
            log.debug("Could not fetch %s: %s", url, exc)
            continue
    log.warning("No tennis-data.co.uk file found for %s %d", tour, year)
    return None


def load_tennis_data_uk(
    tours: list[Literal["atp", "wta"]] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download (or load from cache) all tennis-data.co.uk CSVs.

    Returns a normalised DataFrame with closing ML odds and basic match info.
    Games O/U and Set Spread closing lines are NOT available here — this source
    only provides ML odds.  See reports/phase0_data_gaps.md for the implication.
    """
    if tours is None:
        tours = ["atp", "wta"]

    frames: list[pd.DataFrame] = []

    for tour in tours:
        start, end = _YEAR_RANGES[tour]
        for year in range(start, end + 1):
            cache_path = (
                cache_dir / f"{tour}_{year}.csv" if cache_dir else None
            )
            if cache_path and cache_path.exists():
                df = pd.read_csv(cache_path, low_memory=False)
            else:
                df = _fetch_year(tour, year)
                if df is None:
                    continue
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)

            frames.append(df)
            log.info("Loaded tennis-data.co.uk %s %d — %d rows", tour, year, len(df))

    if not frames:
        raise RuntimeError("No tennis-data.co.uk data loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalise(combined)
    log.info("tennis-data.co.uk total: %d matches", len(combined))
    return combined


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})

    # Parse date
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"]:
        parsed = pd.to_datetime(df["match_date"], format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.8:
            df["match_date"] = parsed
            break
    else:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Numeric coercions
    for col in ["winner_rank", "loser_rank", "best_of",
                "odds_b365_winner", "odds_b365_loser",
                "odds_ps_winner", "odds_ps_loser",
                "odds_avg_winner", "odds_avg_loser"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best available closing odds — prefer Pinnacle (sharpest book), fall back to Avg, then B365
    df["closing_odds_winner"] = df.get("odds_ps_winner", pd.Series(dtype=float))
    df["closing_odds_loser"] = df.get("odds_ps_loser", pd.Series(dtype=float))
    for col_w, col_l in [("odds_avg_winner", "odds_avg_loser"),
                          ("odds_b365_winner", "odds_b365_loser")]:
        if col_w in df.columns:
            df["closing_odds_winner"] = df["closing_odds_winner"].fillna(df[col_w])
            df["closing_odds_loser"] = df["closing_odds_loser"].fillna(df[col_l])

    # Implied probability (no-vig using Pinnacle margin removal)
    df["implied_prob_winner"] = _no_vig_prob(df["closing_odds_winner"], df["closing_odds_loser"])

    df = df.sort_values("match_date").reset_index(drop=True)
    return df


def _no_vig_prob(odds_w: pd.Series, odds_l: pd.Series) -> pd.Series:
    """Pinnacle-style no-vig implied probability for the winner side."""
    raw_w = 1.0 / odds_w
    raw_l = 1.0 / odds_l
    total = raw_w + raw_l
    return raw_w / total


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download tennis-data.co.uk odds data")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/tennis_data_uk"),
    )
    parser.add_argument("--tours", nargs="+", choices=["atp", "wta"], default=["atp", "wta"])
    args = parser.parse_args()

    df = load_tennis_data_uk(tours=args.tours, cache_dir=args.cache_dir)
    print(f"Loaded {len(df):,} matches")
    has_odds = df["closing_odds_winner"].notna().sum()
    print(f"Rows with closing odds: {has_odds:,} ({100*has_odds/len(df):.1f}%)")
    print(df[["match_date", "tour", "winner_name", "loser_name",
              "closing_odds_winner", "closing_odds_loser",
              "implied_prob_winner"]].head(10))
