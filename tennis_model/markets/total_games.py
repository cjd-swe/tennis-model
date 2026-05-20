"""
Total Games Over/Under market.

Outcome: total games played across all sets (integer derived from Sackmann score).
Payoff : given a line (e.g. 22.5) and side ("over" | "under"), determine win/loss.

NOTE: Historical *closing lines* for this market are not available in
tennis-data.co.uk.  In Phase 0 we use this market in price-discovery mode —
comparing model-predicted totals against realised outcomes.  Phase 3 will
source live closing lines from an odds API.

See reports/phase0_data_gaps.md for details.
"""

from __future__ import annotations

import pandas as pd

from tennis_model.markets.base import BetResult, Market


class TotalGamesMarket(Market):
    name = "total_games"

    def outcome(self, row: pd.Series) -> str | None:
        """Return the realised total games as a string (e.g. '23')."""
        total = row.get("total_games")
        if pd.isna(total):
            return None
        # Sanity check — retired matches can have incomplete sets
        score = str(row.get("score", ""))
        if "RET" in score:
            return None  # skip retirements for totals (incomplete contest)
        return str(int(total))

    def payoff(
        self, row: pd.Series, side: str, odds: float, stake: float, line: float = 22.5
    ) -> BetResult:
        """
        Parameters
        ----------
        side : "over" | "under"
        line : the half-game total line (e.g. 22.5 means over needs ≥ 23 games)
        """
        outcome_str = self.outcome(row)
        if outcome_str is None:
            return BetResult(stake=stake, payout=0.0, profit=-stake, won=False, odds=odds,
                             implied_prob=self._implied_prob(odds))
        total = int(outcome_str)
        won = (total > line) if side == "over" else (total < line)
        payout = stake * odds if won else 0.0
        return BetResult(
            stake=stake,
            payout=payout,
            profit=payout - stake,
            won=won,
            odds=odds,
            implied_prob=self._implied_prob(odds),
        )

    @staticmethod
    def realised_total(row: pd.Series) -> int | None:
        """Convenience: just return the integer total."""
        total = row.get("total_games")
        if pd.isna(total):
            return None
        score = str(row.get("score", ""))
        if "RET" in score:
            return None
        return int(total)
