"""
Baseline strategies for Phase 0 profitability investigation.

All strategies operate on the moneyline market.  They serve as:
  - Sanity checks (bet_favorite / bet_underdog should be negative ROI)
  - Performance baselines (ranking_diff, basic_elo, weighted_elo)

Each strategy produces one Pick per test match where odds data is available.
The closing_odds used are `closing_odds_winner` / `closing_odds_loser` from
the merged DataFrame (sourced from Pinnacle where available, else average).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from tennis_model.backtest.engine import Pick
from tennis_model.strategies.base import Strategy

log = logging.getLogger(__name__)

# Minimum odds to consider a match — avoids near-certain chalks
_MIN_ODDS = 1.02
_MAX_ODDS = 15.0   # filter out extreme longshots with data quality risk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_odds_mask(df: pd.DataFrame) -> pd.Series:
    has_w = df["closing_odds_winner"].notna() & df["closing_odds_winner"].between(_MIN_ODDS, _MAX_ODDS)
    has_l = df["closing_odds_loser"].notna() & df["closing_odds_loser"].between(_MIN_ODDS, _MAX_ODDS)
    return has_w & has_l


def _no_vig_prob(odds_w: float, odds_l: float) -> float:
    raw_w = 1.0 / odds_w
    raw_l = 1.0 / odds_l
    return raw_w / (raw_w + raw_l)


# ── Bet Favorite ──────────────────────────────────────────────────────────────

class BetFavorite(Strategy):
    """Always bet the shorter-priced (favorite) side."""
    market = "moneyline"

    def picks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        valid = test_df[_valid_odds_mask(test_df)]
        result = []
        for idx, row in valid.iterrows():
            ow = row["closing_odds_winner"]
            ol = row["closing_odds_loser"]
            if ow <= ol:
                side, odds, model_prob = "winner", ow, _no_vig_prob(ow, ol)
            else:
                side, odds, model_prob = "loser", ol, 1 - _no_vig_prob(ow, ol)
            result.append(Pick(
                match_idx=idx, market=self.market, side=side,
                model_prob=model_prob, closing_odds=odds,
            ))
        return result


# ── Bet Underdog ──────────────────────────────────────────────────────────────

class BetUnderdog(Strategy):
    """Always bet the longer-priced (underdog) side."""
    market = "moneyline"

    def picks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        valid = test_df[_valid_odds_mask(test_df)]
        result = []
        for idx, row in valid.iterrows():
            ow = row["closing_odds_winner"]
            ol = row["closing_odds_loser"]
            if ow >= ol:
                side, odds, model_prob = "winner", ow, _no_vig_prob(ow, ol)
            else:
                side, odds, model_prob = "loser", ol, 1 - _no_vig_prob(ow, ol)
            result.append(Pick(
                match_idx=idx, market=self.market, side=side,
                model_prob=model_prob, closing_odds=odds,
            ))
        return result


# ── Ranking Difference ────────────────────────────────────────────────────────

class RankingDiff(Strategy):
    """
    Bet when the better-ranked player is priced as an underdog — a simple
    market-inefficiency probe based on rank vs. implied prob disagreement.

    Also used as a sanity check: betting the higher-ranked player always should
    roughly match the favorite baseline.
    """
    market = "moneyline"

    def __init__(self, min_edge: float = 0.04, always_bet: bool = False) -> None:
        self.min_edge = min_edge
        self.always_bet = always_bet  # if True, bet higher-ranked player every match

    def picks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        valid = test_df[
            _valid_odds_mask(test_df)
            & test_df["winner_rank"].notna()
            & test_df["loser_rank"].notna()
        ]
        result = []
        for idx, row in valid.iterrows():
            ow = row["closing_odds_winner"]
            ol = row["closing_odds_loser"]
            wr = row["winner_rank"]
            lr = row["loser_rank"]

            # Lower rank number = better ranked
            if wr <= lr:
                # Winner is better ranked
                model_prob = _no_vig_prob(ow, ol)  # rough proxy
                market_implied = 1.0 / ow
                edge = model_prob - market_implied
                if self.always_bet or edge >= self.min_edge:
                    result.append(Pick(
                        match_idx=idx, market=self.market, side="winner",
                        model_prob=model_prob, closing_odds=ow,
                    ))
            else:
                # Loser is better ranked (loser_name on the Sackmann side)
                model_prob = 1 - _no_vig_prob(ow, ol)
                market_implied = 1.0 / ol
                edge = model_prob - market_implied
                if self.always_bet or edge >= self.min_edge:
                    result.append(Pick(
                        match_idx=idx, market=self.market, side="loser",
                        model_prob=model_prob, closing_odds=ol,
                    ))
        return result


# ── Basic Elo ─────────────────────────────────────────────────────────────────

class BasicElo(Strategy):
    """
    Vanilla Elo (k=32) updated match-by-match in training order.
    Bets when Elo-implied probability exceeds market implied probability by
    at least `min_edge`.

    The Elo model is refitted from scratch at the start of each fold using
    train_df, then applied to test_df (no updates during the test window —
    strict out-of-sample).
    """
    market = "moneyline"

    def __init__(self, k: int = 32, initial_rating: float = 1500.0, min_edge: float = 0.03) -> None:
        self.k = k
        self.initial_rating = initial_rating
        self.min_edge = min_edge

    def _fit(self, train_df: pd.DataFrame) -> dict[str, float]:
        ratings: dict[str, float] = {}
        train_sorted = train_df.sort_values("tourney_date")
        for _, row in train_sorted.iterrows():
            w = str(row["winner_name"])
            l = str(row["loser_name"])
            rw = ratings.get(w, self.initial_rating)
            rl = ratings.get(l, self.initial_rating)
            ew = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
            ratings[w] = rw + self.k * (1.0 - ew)
            ratings[l] = rl + self.k * (0.0 - (1.0 - ew))
        return ratings

    def picks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        ratings = self._fit(train_df)
        valid = test_df[_valid_odds_mask(test_df)]
        result = []
        for idx, row in valid.iterrows():
            w = str(row["winner_name"])
            l = str(row["loser_name"])
            rw = ratings.get(w, self.initial_rating)
            rl = ratings.get(l, self.initial_rating)
            elo_prob_w = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))

            ow = row["closing_odds_winner"]
            ol = row["closing_odds_loser"]

            # Pick winner if model thinks they're underpriced
            implied_w = 1.0 / ow
            implied_l = 1.0 / ol
            edge_w = elo_prob_w - implied_w
            edge_l = (1.0 - elo_prob_w) - implied_l

            if edge_w >= self.min_edge and edge_w >= edge_l:
                result.append(Pick(
                    match_idx=idx, market=self.market, side="winner",
                    model_prob=elo_prob_w, closing_odds=ow,
                ))
            elif edge_l >= self.min_edge:
                result.append(Pick(
                    match_idx=idx, market=self.market, side="loser",
                    model_prob=1.0 - elo_prob_w, closing_odds=ol,
                ))
        return result


# ── Surface-Weighted Elo (Angelini et al. 2022) ───────────────────────────────

class WeightedElo(Strategy):
    """
    Maintains three separate Elo ladders (hard, clay, grass/other) and
    blends them based on the upcoming match's surface.

    Blend weights per Angelini et al. 2022: surface-specific Elo gets
    weight `alpha`, overall Elo gets (1 - alpha).
    """
    market = "moneyline"

    SURFACE_MAP = {
        "hard": "hard",
        "clay": "clay",
        "grass": "grass",
        "carpet": "grass",  # treat carpet as grass-adjacent
    }

    def __init__(
        self,
        k: int = 32,
        initial_rating: float = 1500.0,
        alpha: float = 0.5,
        min_edge: float = 0.03,
    ) -> None:
        self.k = k
        self.initial_rating = initial_rating
        self.alpha = alpha
        self.min_edge = min_edge

    def _fit(self, train_df: pd.DataFrame) -> tuple[dict, dict]:
        overall: dict[str, float] = {}
        surface_ratings: dict[str, dict[str, float]] = {
            "hard": {}, "clay": {}, "grass": {}
        }
        train_sorted = train_df.sort_values("tourney_date")
        for _, row in train_sorted.iterrows():
            w = str(row["winner_name"])
            l = str(row["loser_name"])
            surf = self.SURFACE_MAP.get(str(row.get("surface", "")).lower(), "hard")

            # Update overall
            rw_o = overall.get(w, self.initial_rating)
            rl_o = overall.get(l, self.initial_rating)
            ew_o = 1.0 / (1.0 + 10 ** ((rl_o - rw_o) / 400.0))
            overall[w] = rw_o + self.k * (1.0 - ew_o)
            overall[l] = rl_o + self.k * (0.0 - (1.0 - ew_o))

            # Update surface-specific
            sr = surface_ratings[surf]
            rw_s = sr.get(w, self.initial_rating)
            rl_s = sr.get(l, self.initial_rating)
            ew_s = 1.0 / (1.0 + 10 ** ((rl_s - rw_s) / 400.0))
            sr[w] = rw_s + self.k * (1.0 - ew_s)
            sr[l] = rl_s + self.k * (0.0 - (1.0 - ew_s))

        return overall, surface_ratings

    def picks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        overall, surface_ratings = self._fit(train_df)
        valid = test_df[_valid_odds_mask(test_df)]
        result = []
        for idx, row in valid.iterrows():
            w = str(row["winner_name"])
            l = str(row["loser_name"])
            surf = self.SURFACE_MAP.get(str(row.get("surface", "")).lower(), "hard")
            sr = surface_ratings[surf]

            rw_o = overall.get(w, self.initial_rating)
            rl_o = overall.get(l, self.initial_rating)
            rw_s = sr.get(w, self.initial_rating)
            rl_s = sr.get(l, self.initial_rating)

            # Blended rating
            rw = self.alpha * rw_s + (1 - self.alpha) * rw_o
            rl = self.alpha * rl_s + (1 - self.alpha) * rl_o
            elo_prob_w = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))

            ow = row["closing_odds_winner"]
            ol = row["closing_odds_loser"]
            implied_w = 1.0 / ow
            implied_l = 1.0 / ol
            edge_w = elo_prob_w - implied_w
            edge_l = (1.0 - elo_prob_w) - implied_l

            if edge_w >= self.min_edge and edge_w >= edge_l:
                result.append(Pick(
                    match_idx=idx, market=self.market, side="winner",
                    model_prob=elo_prob_w, closing_odds=ow,
                ))
            elif edge_l >= self.min_edge:
                result.append(Pick(
                    match_idx=idx, market=self.market, side="loser",
                    model_prob=1.0 - elo_prob_w, closing_odds=ol,
                ))
        return result
