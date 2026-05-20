"""
Base class for betting strategies.

A strategy must implement picks(train_df, test_df) -> list[Pick].
The test_df contains only the rows for the out-of-sample window;
train_df contains all rows strictly before the test window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from tennis_model.backtest.engine import Pick


class Strategy(ABC):
    market: str  # "moneyline" | "total_games" | "set_spread"

    @abstractmethod
    def picks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        """Return a list of Picks for the test window."""

    def __call__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[Pick]:
        return self.picks(train_df, test_df)
