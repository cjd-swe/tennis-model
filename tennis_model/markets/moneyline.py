"""
Moneyline (match winner) market.

Outcome: "winner" | "loser" (relative to Sackmann convention — winner_name won).
Odds  : closing_odds_winner / closing_odds_loser from tennis-data.co.uk.
"""

from __future__ import annotations

import pandas as pd

from tennis_model.markets.base import BetResult, Market


class MoneylineMarket(Market):
    name = "moneyline"

    def outcome(self, row: pd.Series) -> str | None:
        # Sackmann always has winner_name win; any retirement/walkover still
        # resolves (the retiring player is the loser).  Only skip if score is NaN.
        if pd.isna(row.get("score")) and pd.isna(row.get("total_games")):
            return None
        # Filter walkovers — no meaningful contest
        score = str(row.get("score", ""))
        if "W/O" in score or "walkover" in score.lower():
            return None
        return "winner"  # winner_name always wins

    def payoff(self, row: pd.Series, side: str, odds: float, stake: float) -> BetResult:
        actual = self.outcome(row)
        if actual is None:
            return BetResult(stake=stake, payout=0.0, profit=-stake, won=False, odds=odds,
                             implied_prob=self._implied_prob(odds))
        won = side == actual
        payout = stake * odds if won else 0.0
        return BetResult(
            stake=stake,
            payout=payout,
            profit=payout - stake,
            won=won,
            odds=odds,
            implied_prob=self._implied_prob(odds),
        )
