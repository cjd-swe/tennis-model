"""Base class for all market definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd


Side = Literal["winner", "loser"]


@dataclass
class BetResult:
    """Outcome of a single evaluated bet."""
    stake: float
    payout: float          # stake returned + profit; 0 if lost
    profit: float          # payout - stake
    won: bool
    odds: float            # decimal odds at which the bet was taken
    implied_prob: float    # 1/odds (before no-vig adjustment)


class Market(ABC):
    """
    A betting market type.  Subclasses define how outcomes are determined
    and how payoffs are computed from decimal odds.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def outcome(self, row: pd.Series) -> str | None:
        """
        Return a string label for the realised outcome of this match row.
        Return None if outcome cannot be determined (retired, walkover, etc.).
        """

    @abstractmethod
    def payoff(self, row: pd.Series, side: str, odds: float, stake: float) -> BetResult:
        """
        Compute the result of betting `side` at `odds` (decimal) for `stake`.
        """

    @staticmethod
    def _implied_prob(decimal_odds: float) -> float:
        if decimal_odds <= 0:
            return float("nan")
        return 1.0 / decimal_odds
