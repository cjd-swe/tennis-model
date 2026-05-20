"""
Walk-forward backtesting engine.

Design:
- Calendar-based folds: train on rolling N years, test on next K months.
- No-leakage contract: every strategy/feature declares a lookback window;
  the engine enforces that test rows only see data with match_date < fold_start.
- CLV (closing-line value) tracking: measures edge vs. the closing market.
- Bankroll simulation: flat stake and fractional Kelly, with drawdown tracking.
- Subsegment slicing built in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal

import numpy as np
import pandas as pd

from tennis_model.markets.base import BetResult, Market

log = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Pick:
    """A model's pick for a single match."""
    match_idx: int           # index in the original DataFrame
    market: str              # "moneyline" | "total_games" | "set_spread"
    side: str                # e.g. "winner", "over", "loser"
    model_prob: float        # model's estimated win probability for `side`
    closing_odds: float      # decimal odds at closing (from the market)
    line: float | None = None  # for O/U / spread


@dataclass
class BetRecord:
    """Evaluated outcome of a Pick."""
    pick: Pick
    result: BetResult
    fold_id: int
    match_date: pd.Timestamp
    surface: str | None
    tour: str | None
    round_: str | None
    best_of: int | None
    winner_rank: float | None
    loser_rank: float | None
    # CLV: closing implied prob for our side vs. our model prob
    clv: float | None  # closing_implied_prob - model_prob  (positive = we were right before close)


@dataclass
class BacktestConfig:
    train_years: int = 3           # rolling training window length
    test_months: int = 6           # out-of-sample window per fold
    min_train_rows: int = 500      # skip folds with insufficient history
    flat_stake: float = 1.0        # units per bet for flat-stake sim
    kelly_fraction: float = 0.25   # fractional Kelly multiplier
    min_edge: float = 0.02         # minimum model_prob - implied_prob to place a bet
    max_kelly_stake: float = 5.0   # cap stake at N units for bankroll safety


# ── Walk-forward fold generator ───────────────────────────────────────────────

def _fold_dates(
    df: pd.DataFrame,
    date_col: str,
    train_years: int,
    test_months: int,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Yield (train_start, fold_start, fold_end) for each walk-forward fold.
    First fold starts when there are at least `train_years` years of data.
    """
    dates = df[date_col].dropna().sort_values()
    overall_start = dates.iloc[0]
    overall_end = dates.iloc[-1]

    fold_start = overall_start + pd.DateOffset(years=train_years)
    fold_start = fold_start.replace(day=1)  # align to month boundary

    while fold_start < overall_end:
        fold_end = fold_start + pd.DateOffset(months=test_months)
        train_start = fold_start - pd.DateOffset(years=train_years)
        yield train_start, fold_start, min(fold_end, overall_end)
        fold_start = fold_end


# ── CLV computation ───────────────────────────────────────────────────────────

def _compute_clv(model_prob: float, closing_odds: float) -> float | None:
    """
    CLV = closing implied prob (no-vig) - model_prob.
    Positive means closing market moved toward us → we had early edge.
    """
    if closing_odds <= 1.0 or np.isnan(closing_odds):
        return None
    closing_implied = 1.0 / closing_odds
    return closing_implied - model_prob


# ── Kelly stake ───────────────────────────────────────────────────────────────

def _kelly_stake(
    model_prob: float,
    decimal_odds: float,
    fraction: float,
    max_stake: float,
    flat_stake: float,
) -> float:
    b = decimal_odds - 1.0  # net odds
    q = 1.0 - model_prob
    kelly = (model_prob * b - q) / b
    stake = min(max(kelly * fraction, 0.0), max_stake) * flat_stake
    return stake


# ── Main backtester ───────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester.

    Parameters
    ----------
    df : merged match DataFrame (from tennis_model.ingest.merge)
    market : Market instance defining outcome and payoff logic
    strategy : callable(train_df, test_df) -> list[Pick]
        The strategy must only use information available before each test match.
        It receives the full training slice and the test slice for picking.
    config : BacktestConfig
    date_col : name of the match date column in df
    """

    def __init__(
        self,
        df: pd.DataFrame,
        market: Market,
        strategy: Callable[[pd.DataFrame, pd.DataFrame], list[Pick]],
        config: BacktestConfig | None = None,
        date_col: str = "tourney_date",
    ) -> None:
        self.df = df.copy()
        self.market = market
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.date_col = date_col

        # Validate date column
        if date_col not in self.df.columns:
            raise ValueError(f"date_col '{date_col}' not in DataFrame")
        self.df[date_col] = pd.to_datetime(self.df[date_col])

    def run(self) -> BacktestResult:
        records: list[BetRecord] = []
        cfg = self.config

        for fold_id, (train_start, fold_start, fold_end) in enumerate(
            _fold_dates(self.df, self.date_col, cfg.train_years, cfg.test_months)
        ):
            train_mask = (
                (self.df[self.date_col] >= train_start)
                & (self.df[self.date_col] < fold_start)
            )
            test_mask = (
                (self.df[self.date_col] >= fold_start)
                & (self.df[self.date_col] < fold_end)
            )

            train_df = self.df[train_mask].copy()
            test_df = self.df[test_mask].copy()

            if len(train_df) < cfg.min_train_rows:
                log.debug("Fold %d: insufficient train rows (%d), skipping", fold_id, len(train_df))
                continue

            if len(test_df) == 0:
                continue

            log.debug(
                "Fold %d: train %s–%s (%d rows), test %s–%s (%d rows)",
                fold_id,
                train_start.date(), fold_start.date(), len(train_df),
                fold_start.date(), fold_end.date(), len(test_df),
            )

            picks = self.strategy(train_df, test_df)

            for pick in picks:
                row = self.df.loc[pick.match_idx]
                result = self.market.payoff(
                    row,
                    side=pick.side,
                    odds=pick.closing_odds,
                    stake=cfg.flat_stake,
                    **({} if pick.line is None else {"line": pick.line}),
                )
                clv = _compute_clv(pick.model_prob, pick.closing_odds)

                record = BetRecord(
                    pick=pick,
                    result=result,
                    fold_id=fold_id,
                    match_date=row.get(self.date_col),
                    surface=row.get("surface"),
                    tour=row.get("tour"),
                    round_=row.get("round"),
                    best_of=row.get("best_of"),
                    winner_rank=row.get("winner_rank"),
                    loser_rank=row.get("loser_rank"),
                    clv=clv,
                )
                records.append(record)

        return BacktestResult(records=records, config=cfg, market_name=self.market.name)


# ── Results container ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    records: list[BetRecord]
    config: BacktestConfig
    market_name: str

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({
                "fold_id": r.fold_id,
                "match_date": r.match_date,
                "market": r.pick.market,
                "side": r.pick.side,
                "model_prob": r.pick.model_prob,
                "closing_odds": r.pick.closing_odds,
                "closing_implied_prob": 1.0 / r.pick.closing_odds if r.pick.closing_odds > 1 else None,
                "edge": r.pick.model_prob - (1.0 / r.pick.closing_odds) if r.pick.closing_odds > 1 else None,
                "clv": r.clv,
                "stake": r.result.stake,
                "profit": r.result.profit,
                "won": r.result.won,
                "odds": r.result.odds,
                "surface": r.surface,
                "tour": r.tour,
                "round": r.round_,
                "best_of": r.best_of,
                "winner_rank": r.winner_rank,
                "loser_rank": r.loser_rank,
                "rank_diff": (
                    (r.winner_rank - r.loser_rank)
                    if r.winner_rank and r.loser_rank else None
                ),
            })
        return pd.DataFrame(rows)

    def summary(self, df: pd.DataFrame | None = None) -> dict:
        """Compute overall summary statistics."""
        if df is None:
            df = self.to_dataframe()
        if len(df) == 0:
            return {"n_bets": 0}

        total_stake = df["stake"].sum()
        total_profit = df["profit"].sum()
        roi = total_profit / total_stake if total_stake > 0 else float("nan")

        # Bankroll simulation (flat stake, running cumulative)
        df_sorted = df.sort_values("match_date")
        cumulative = df_sorted["profit"].cumsum()
        max_dd = _max_drawdown(cumulative.values)

        return {
            "n_bets": len(df),
            "hit_rate": df["won"].mean(),
            "avg_odds": df["odds"].mean(),
            "total_profit": total_profit,
            "roi": roi,
            "clv_mean": df["clv"].mean() if "clv" in df else None,
            "clv_pct_positive": (df["clv"] > 0).mean() if "clv" in df else None,
            "max_drawdown": max_dd,
            "sharpe": _sharpe(df_sorted["profit"].values),
        }

    def slice_summary(
        self,
        by: str | list[str],
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Summarise by subsegment (surface, tour, round, best_of, ...)."""
        if df is None:
            df = self.to_dataframe()
        by = [by] if isinstance(by, str) else by

        def _agg(grp: pd.DataFrame) -> pd.Series:
            total_stake = grp["stake"].sum()
            total_profit = grp["profit"].sum()
            return pd.Series({
                "n_bets": len(grp),
                "hit_rate": grp["won"].mean(),
                "roi": total_profit / total_stake if total_stake > 0 else float("nan"),
                "clv_mean": grp["clv"].mean(),
                "avg_odds": grp["odds"].mean(),
            })

        return df.groupby(by).apply(_agg).reset_index()


def _max_drawdown(cumulative: np.ndarray) -> float:
    if len(cumulative) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return float(drawdown.min())


def _sharpe(profits: np.ndarray, risk_free: float = 0.0) -> float:
    if len(profits) < 2:
        return float("nan")
    excess = profits - risk_free
    return float(excess.mean() / excess.std(ddof=1)) if excess.std(ddof=1) > 0 else float("nan")
