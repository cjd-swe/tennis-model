"""
Tests for the walk-forward backtesting engine.

Key invariants:
1. The backtester must not allow any test match data to appear in train_df.
2. CLV is computed correctly against known odds.
3. Strategy sees only past data.
4. Elo computed on hand-known sequence matches expected ratings.
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from tennis_model.backtest.engine import (
    Backtester,
    BacktestConfig,
    Pick,
    _compute_clv,
    _max_drawdown,
)
from tennis_model.markets.moneyline import MoneylineMarket
from tennis_model.strategies.baselines import BasicElo


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, start_year: int = 2010) -> pd.DataFrame:
    """Synthetic match DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=f"{start_year}-01-01", periods=n, freq="2D")
    df = pd.DataFrame({
        "tourney_date": dates,
        "winner_name": [f"PlayerA_{i % 10}" for i in range(n)],
        "loser_name": [f"PlayerB_{i % 10}" for i in range(n)],
        "score": ["6-4 6-2"] * n,
        "total_games": [18] * n,
        "winner_sets": [2] * n,
        "loser_sets": [0] * n,
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tour": ["atp"] * n,
        "round": ["R32"] * n,
        "best_of": [3] * n,
        "winner_rank": rng.integers(1, 200, n).astype(float),
        "loser_rank": rng.integers(1, 200, n).astype(float),
        "closing_odds_winner": rng.uniform(1.3, 3.5, n),
        "closing_odds_loser": None,  # filled below
    })
    df["closing_odds_loser"] = 1.0 / (1.0 - 1.0 / df["closing_odds_winner"]) * 1.04  # rough
    df["closing_odds_loser"] = df["closing_odds_loser"].clip(1.05, 8.0)
    return df


# ── No-leakage test ───────────────────────────────────────────────────────────

class LeakageDetectorStrategy:
    """Raises if any test-window date appears in train_df."""
    market = "moneyline"

    def __call__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        test_dates = set(test_df["tourney_date"])
        leaked = set(train_df["tourney_date"]) & test_dates
        assert not leaked, f"Leakage detected: {leaked}"
        return []  # no picks needed, just asserting


def test_no_leakage():
    df = _make_df(n=200)
    bt = Backtester(
        df=df,
        market=MoneylineMarket(),
        strategy=LeakageDetectorStrategy(),
        config=BacktestConfig(train_years=1, test_months=3, min_train_rows=10),
    )
    bt.run()  # would raise if leakage detected


# ── CLV computation ───────────────────────────────────────────────────────────

def test_clv_positive_when_market_moved_toward_us():
    # Model prob 0.60, closing odds 1.60 → closing implied = 0.625
    # CLV = 0.625 - 0.60 = 0.025 (positive — market agrees we were right)
    clv = _compute_clv(model_prob=0.60, closing_odds=1.60)
    assert clv is not None
    assert abs(clv - 0.025) < 1e-9


def test_clv_negative_when_market_moved_against_us():
    clv = _compute_clv(model_prob=0.70, closing_odds=1.80)
    # closing implied = 1/1.80 ≈ 0.556
    assert clv is not None
    assert clv < 0


def test_clv_invalid_odds():
    assert _compute_clv(0.5, 0.0) is None
    assert _compute_clv(0.5, float("nan")) is None


# ── Max drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_simple():
    cumulative = np.array([0, 1, 2, 1, 0, -1, 0])
    assert _max_drawdown(cumulative) == -3.0  # peak=2, trough=-1


def test_max_drawdown_monotone_increase():
    assert _max_drawdown(np.array([0, 1, 2, 3])) == 0.0


# ── Elo hand-computation ──────────────────────────────────────────────────────

def test_basic_elo_hand_computed():
    """
    Verify BasicElo updates match manual calculation.
    Match: PlayerA (1500) beats PlayerB (1500), k=32.
    Expected P(A wins) = 0.5 → after win: A=1516, B=1484.
    """
    elo = BasicElo(k=32, initial_rating=1500.0, min_edge=0.0)
    train = pd.DataFrame({
        "tourney_date": [pd.Timestamp("2010-01-01")],
        "winner_name": ["PlayerA"],
        "loser_name": ["PlayerB"],
    })
    ratings = elo._fit(train)
    assert abs(ratings["PlayerA"] - 1516.0) < 0.1
    assert abs(ratings["PlayerB"] - 1484.0) < 0.1


# ── End-to-end smoke test ─────────────────────────────────────────────────────

def test_backtester_produces_records():
    df = _make_df(n=300)
    elo = BasicElo(k=32, min_edge=0.0)  # min_edge=0 so we get picks
    bt = Backtester(
        df=df,
        market=MoneylineMarket(),
        strategy=elo,
        config=BacktestConfig(train_years=1, test_months=3, min_train_rows=50),
    )
    result = bt.run()
    assert len(result.records) > 0
    summary = result.summary()
    assert "roi" in summary
    assert "n_bets" in summary
    assert summary["n_bets"] == len(result.records)
