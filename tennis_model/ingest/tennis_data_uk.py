"""
Download historical match + closing-odds Excel files from tennis-data.co.uk.

Actual URL structure (confirmed from alldata.php):
  ATP:  http://www.tennis-data.co.uk/{year}/{year}.xls   (2000–2012)
        http://www.tennis-data.co.uk/{year}/{year}.xlsx  (2013+)
  WTA:  http://www.tennis-data.co.uk/{year}w/{year}.xls  (2007–2012)
        http://www.tennis-data.co.uk/{year}w/{year}.xlsx (2013+)

There are no CSV files — the site uses Excel only.

Odds columns:
  B365W/B365L — Bet365 winner/loser ML odds
  PSW/PSL     — Pinnacle winner/loser ML odds
  AvgW/AvgL   — market-average winner/loser ML odds

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

_BASE = "http://www.tennis-data.co.uk"

_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "atp": (2000, 2026),
    "wta": (2007, 2026),
}

# 2013 is the transition year from .xls → .xlsx
_XLS_TO_XLSX_YEAR = 2013

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


def _file_url(tour: Literal["atp", "wta"], year: int) -> str:
    ext = ".xlsx" if year >= _XLS_TO_XLSX_YEAR else ".xls"
    directory = f"{year}w" if tour == "wta" else str(year)
    return f"{_BASE}/{directory}/{year}{ext}"


def _fetch_year(tour: Literal["atp", "wta"], year: int) -> pd.DataFrame | None:
    url = _file_url(tour, year)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            log.debug("HTTP %d for %s", resp.status_code, url)
            return None
        # Verify it's actually an Excel file (PK magic bytes for xlsx, D0CF for xls)
        magic = resp.content[:4]
        if magic[:2] not in (b"PK", b"\xd0\xcf") and magic[:2] != b"PK":
            # Server returned HTML (e.g. a 300 page with 200 status)
            log.debug("Non-Excel response for %s (got %r)", url, magic)
            return None
        ext = ".xlsx" if year >= _XLS_TO_XLSX_YEAR else ".xls"
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        df = pd.read_excel(io.BytesIO(resp.content), engine=engine)
        df["tour"] = tour
        df["year"] = year
        return df
    except Exception as exc:
        log.debug("Could not fetch %s: %s", url, exc)
        return None


def load_tennis_data_uk(
    tours: list[Literal["atp", "wta"]] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download (or load from cache) all tennis-data.co.uk Excel files.

    Returns a normalised DataFrame with closing ML odds and basic match info.
    Games O/U and Set Spread closing lines are NOT available from this source.
    See reports/phase0_data_gaps.md.
    """
    if tours is None:
        tours = ["atp", "wta"]

    frames: list[pd.DataFrame] = []

    for tour in tours:
        start, end = _YEAR_RANGES[tour]
        for year in range(start, end + 1):
            cache_path = cache_dir / f"{tour}_{year}.parquet" if cache_dir else None
            if cache_path and cache_path.exists():
                df = pd.read_parquet(cache_path)
            else:
                raw = _fetch_year(tour, year)
                if raw is None:
                    log.warning("No tennis-data.co.uk file found for %s %d", tour, year)
                    continue
                # Normalise before caching so type coercions run first
                # (raw files contain 'NR' strings in rank columns etc.)
                df = _normalise(raw)
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_path, index=False)

            frames.append(df)
            log.info("Loaded tennis-data.co.uk %s %d — %d rows", tour, year, len(df))

    if not frames:
        raise RuntimeError("No tennis-data.co.uk data loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("match_date").reset_index(drop=True)
    log.info("tennis-data.co.uk total: %d matches", len(combined))
    return combined


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})

    # Parse date — try common formats in order of likelihood
    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"]:
        parsed = pd.to_datetime(df["match_date"], format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.8:
            df["match_date"] = parsed
            break
    else:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Coerce all columns that should be numeric (handles 'NR', ' ', blanks, etc.)
    _NUMERIC_COLS = [
        "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
        "best_of", "winner_sets", "loser_sets",
        "w_s1", "w_s2", "w_s3", "w_s4", "w_s5",
        "l_s1", "l_s2", "l_s3", "l_s4", "l_s5",
        "odds_b365_winner", "odds_b365_loser",
        "odds_ps_winner", "odds_ps_loser",
        "odds_avg_winner", "odds_avg_loser",
        "odds_max_winner", "odds_max_loser",
    ]
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best closing odds: prefer Pinnacle (sharpest), fall back to Avg, then B365
    df["closing_odds_winner"] = df.get("odds_ps_winner", pd.Series(dtype=float))
    df["closing_odds_loser"] = df.get("odds_ps_loser", pd.Series(dtype=float))
    for col_w, col_l in [("odds_avg_winner", "odds_avg_loser"),
                          ("odds_b365_winner", "odds_b365_loser")]:
        if col_w in df.columns:
            df["closing_odds_winner"] = df["closing_odds_winner"].fillna(df[col_w])
            df["closing_odds_loser"] = df["closing_odds_loser"].fillna(df[col_l])

    df["implied_prob_winner"] = _no_vig_prob(df["closing_odds_winner"], df["closing_odds_loser"])

    # Cast remaining object columns to string so Parquet serialisation never fails
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace({"nan": None, "<NA>": None, " ": None})

    df = df.sort_values("match_date").reset_index(drop=True)
    return df


def _no_vig_prob(odds_w: pd.Series, odds_l: pd.Series) -> pd.Series:
    raw_w = 1.0 / odds_w
    raw_l = 1.0 / odds_l
    total = raw_w + raw_l
    return raw_w / total


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download tennis-data.co.uk odds data")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/raw/tennis_data_uk"),
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
