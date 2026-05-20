"""Unit tests for market outcome and payoff logic."""

import pandas as pd
import pytest

from tennis_model.markets.moneyline import MoneylineMarket
from tennis_model.markets.total_games import TotalGamesMarket
from tennis_model.markets.set_spread import SetSpreadMarket


def _row(**kwargs) -> pd.Series:
    defaults = {
        "score": "6-4 6-2",
        "total_games": 18,
        "winner_sets": 2,
        "loser_sets": 0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestMoneyline:
    m = MoneylineMarket()

    def test_winner_outcome(self):
        assert self.m.outcome(_row()) == "winner"

    def test_walkover_excluded(self):
        assert self.m.outcome(_row(score="W/O")) is None

    def test_bet_winner_correct(self):
        r = self.m.payoff(_row(), side="winner", odds=1.5, stake=10.0)
        assert r.won is True
        assert abs(r.profit - 5.0) < 1e-9

    def test_bet_loser_wrong(self):
        r = self.m.payoff(_row(), side="loser", odds=3.0, stake=10.0)
        assert r.won is False
        assert r.profit == -10.0


class TestTotalGames:
    m = TotalGamesMarket()

    def test_outcome_normal(self):
        assert self.m.outcome(_row(score="6-4 6-2", total_games=18)) == "18"

    def test_retirement_excluded(self):
        assert self.m.outcome(_row(score="6-4 RET", total_games=10)) is None

    def test_over_wins(self):
        r = self.m.payoff(_row(total_games=23), side="over", odds=1.9, stake=10.0, line=22.5)
        assert r.won is True

    def test_under_wins(self):
        r = self.m.payoff(_row(total_games=20), side="under", odds=1.9, stake=10.0, line=22.5)
        assert r.won is True

    def test_over_loses(self):
        r = self.m.payoff(_row(total_games=20), side="over", odds=1.9, stake=10.0, line=22.5)
        assert r.won is False


class TestSetSpread:
    m = SetSpreadMarket()

    def test_outcome_2_0(self):
        assert self.m.outcome(_row(score="6-4 6-2", winner_sets=2, loser_sets=0)) == "+2"

    def test_outcome_2_1(self):
        assert self.m.outcome(_row(score="6-4 3-6 6-3", winner_sets=2, loser_sets=1)) == "+1"

    def test_retirement_excluded(self):
        assert self.m.outcome(_row(score="6-4 RET", winner_sets=1, loser_sets=0)) is None

    def test_winner_minus_1_5_covers_2_0(self):
        r = self.m.payoff(
            _row(score="6-4 6-2", winner_sets=2, loser_sets=0),
            side="winner", odds=1.7, stake=10.0, spread=-1.5,
        )
        assert r.won is True   # margin=2, 2-1.5=0.5>0

    def test_winner_minus_1_5_loses_2_1(self):
        r = self.m.payoff(
            _row(score="6-4 3-6 6-3", winner_sets=2, loser_sets=1),
            side="winner", odds=1.7, stake=10.0, spread=-1.5,
        )
        assert r.won is False  # margin=1, 1-1.5=-0.5<0

    def test_loser_plus_1_5_covers_2_1(self):
        r = self.m.payoff(
            _row(score="6-4 3-6 6-3", winner_sets=2, loser_sets=1),
            side="loser", odds=2.1, stake=10.0, spread=-1.5,
        )
        assert r.won is True   # loser side: margin+spread = 1-1.5=-0.5 < 0 → covers
