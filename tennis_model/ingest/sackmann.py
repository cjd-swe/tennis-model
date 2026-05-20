"""
Pull Jeff Sackmann's tennis_atp / tennis_wta match CSV files and combine
them into a single unified DataFrame.

Usage (CLI):
    uv run python -m tennis_model.ingest.sackmann
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── Column renames to a shared internal schema ──────────────────────────────
_RENAME = {
    "tourney_id": "tourney_id",
    "tourney_name": "tourney_name",
    "surface": "surface",
    "draw_size": "draw_size",
    "tourney_level": "tourney_level",
    "tourney_date": "tourney_date",
    "match_num": "match_num",
    "winner_id": "winner_id",
    "winner_seed": "winner_seed",
    "winner_entry": "winner_entry",
    "winner_name": "winner_name",
    "winner_hand": "winner_hand",
    "winner_ht": "winner_ht",
    "winner_ioc": "winner_ioc",
    "winner_age": "winner_age",
    "winner_rank": "winner_rank",
    "winner_rank_points": "winner_rank_points",
    "loser_id": "loser_id",
    "loser_seed": "loser_seed",
    "loser_entry": "loser_entry",
    "loser_name": "loser_name",
    "loser_hand": "loser_hand",
    "loser_ht": "loser_ht",
    "loser_ioc": "loser_ioc",
    "loser_age": "loser_age",
    "loser_rank": "loser_rank",
    "loser_rank_points": "loser_rank_points",
    "score": "score",
    "best_of": "best_of",
    "round": "round",
    "minutes": "minutes",
    # serve / return stats
    "w_ace": "w_ace",
    "w_df": "w_df",
    "w_svpt": "w_svpt",
    "w_1stIn": "w_1stIn",
    "w_1stWon": "w_1stWon",
    "w_2ndWon": "w_2ndWon",
    "w_SvGms": "w_SvGms",
    "w_bpSaved": "w_bpSaved",
    "w_bpFaced": "w_bpFaced",
    "l_ace": "l_ace",
    "l_df": "l_df",
    "l_svpt": "l_svpt",
    "l_1stIn": "l_1stIn",
    "l_1stWon": "l_1stWon",
    "l_2ndWon": "l_2ndWon",
    "l_SvGms": "l_SvGms",
    "l_bpSaved": "l_bpSaved",
    "l_bpFaced": "l_bpFaced",
}

_GITHUB_RAW = "https://raw.githubusercontent.com/JeffSackmann/{repo}/master"
_ATP_REPO = "tennis_atp"
_WTA_REPO = "tennis_wta"

# ATP odds data starts in 2001; WTA in 2007 — match Sackmann year range to that
_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "atp": (2001, 2024),
    "wta": (2007, 2024),
}


def _fetch_year(tour: Literal["atp", "wta"], year: int) -> pd.DataFrame | None:
    repo = _ATP_REPO if tour == "atp" else _WTA_REPO
    prefix = tour
    url = f"{_GITHUB_RAW.format(repo=repo)}/{prefix}_matches_{year}.csv"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        log.debug("No Sackmann file for %s %d", tour, year)
        return None
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    df["tour"] = tour
    df["year"] = year
    return df


def load_sackmann(
    tours: list[Literal["atp", "wta"]] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download (or load from cache) all Sackmann match CSVs for the given tours.

    Parameters
    ----------
    tours : list of "atp" | "wta", default both
    cache_dir : if given, CSVs are cached here and re-read on subsequent calls

    Returns
    -------
    DataFrame with a unified schema, sorted by tourney_date.
    """
    if tours is None:
        tours = ["atp", "wta"]

    frames: list[pd.DataFrame] = []

    for tour in tours:
        start, end = _YEAR_RANGES[tour]
        for year in range(start, end + 1):
            cache_path = (
                cache_dir / f"{tour}_matches_{year}.csv" if cache_dir else None
            )
            if cache_path and cache_path.exists():
                df = pd.read_csv(cache_path, low_memory=False)
                df["tour"] = tour
                df["year"] = year
            else:
                df = _fetch_year(tour, year)
                if df is None:
                    continue
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False)

            frames.append(df)
            log.info("Loaded %s %d — %d rows", tour, year, len(df))

    if not frames:
        raise RuntimeError("No Sackmann data loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # ── Normalise date to datetime ──────────────────────────────────────────
    combined["tourney_date"] = pd.to_datetime(
        combined["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # ── Parse set scores → total games ─────────────────────────────────────
    combined["total_games"] = combined["score"].apply(_score_to_total_games)
    combined["winner_sets"] = combined["score"].apply(_count_winner_sets)
    combined["loser_sets"] = combined["score"].apply(_count_loser_sets)

    # ── Numeric coercions ───────────────────────────────────────────────────
    for col in ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points",
                "best_of", "w_ace", "l_ace", "w_df", "l_df", "minutes"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    log.info("Sackmann total: %d matches across %s", len(combined), tours)
    return combined


# ── Score parsing helpers ────────────────────────────────────────────────────

def _parse_sets(score: str) -> list[tuple[int, int]]:
    """
    Parse a Sackmann score string into a list of (winner_games, loser_games) tuples.
    Handles retirements, walkovers, and tiebreak notation.
    """
    if not isinstance(score, str):
        return []
    # strip retirement / walkover suffix
    for suffix in [" RET", " W/O", " DEF", " ABN", " UNP"]:
        score = score.replace(suffix, "")
    sets = []
    for part in score.strip().split():
        # e.g. "6-4", "7-6(3)", "1-0"
        part = part.split("(")[0]  # drop tiebreak score
        nums = part.split("-")
        if len(nums) == 2:
            try:
                sets.append((int(nums[0]), int(nums[1])))
            except ValueError:
                pass
    return sets


def _score_to_total_games(score: str) -> int | None:
    sets = _parse_sets(score)
    if not sets:
        return None
    return sum(a + b for a, b in sets)


def _count_winner_sets(score: str) -> int | None:
    sets = _parse_sets(score)
    if not sets:
        return None
    return sum(1 for a, b in sets if a > b)


def _count_loser_sets(score: str) -> int | None:
    sets = _parse_sets(score)
    if not sets:
        return None
    return sum(1 for a, b in sets if b > a)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download Sackmann tennis data")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/sackmann"),
        help="Directory to cache downloaded CSVs (default: data/raw/sackmann)",
    )
    parser.add_argument(
        "--tours",
        nargs="+",
        choices=["atp", "wta"],
        default=["atp", "wta"],
    )
    args = parser.parse_args()

    df = load_sackmann(tours=args.tours, cache_dir=args.cache_dir)
    print(f"Loaded {len(df):,} matches")
    print(df[["tourney_date", "tour", "winner_name", "loser_name", "score", "total_games"]].head(10))
