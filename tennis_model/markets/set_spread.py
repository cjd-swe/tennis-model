"""
Set Spread (set handicap) market.

Standard lines:
  - Best-of-3: winner -1.5 / loser +1.5
    → winner -1.5 wins if they win 2-0; loser +1.5 wins if they win 2-1 or 0-2
  - Best-of-5: winner -2.5 / loser +2.5, winner -1.5 / loser +1.5

Outcome: set score margin from the winner's perspective (positive integer).
  e.g. 2-0 → +2, 2-1 → +1, 3-1 → +2, 3-2 → +1

NOTE: Closing lines for this market are not available in tennis-data.co.uk.
Price-discovery mode only in Phase 0.  See reports/phase0_data_gaps.md.
"""

from __future__ import annotations

import pandas as pd

from tennis_model.markets.base import BetResult, Market


class SetSpreadMarket(Market):
    name = "set_spread"

    def outcome(self, row: pd.Series) -> str | None:
        """
        Return set-score margin as a string, e.g. "+2" (winner won by 2 sets).
        None for retirements and walkovers.
        """
        score = str(row.get("score", ""))
        if "RET" in score or "W/O" in score:
            return None
        w_sets = row.get("winner_sets")
        l_sets = row.get("loser_sets")
        if pd.isna(w_sets) or pd.isna(l_sets):
            return None
        margin = int(w_sets) - int(l_sets)
        return f"+{margin}" if margin > 0 else str(margin)

    def payoff(
        self,
        row: pd.Series,
        side: str,
        odds: float,
        stake: float,
        spread: float = -1.5,
    ) -> BetResult:
        """
        Parameters
        ----------
        side    : "winner" | "loser" (relative to Sackmann winner_name)
        spread  : handicap applied to the winner side (e.g. -1.5 means winner
                  must win by 2+ sets; +1.5 means loser side covers if they
                  win any set or take the match)
        """
        outcome_str = self.outcome(row)
        if outcome_str is None:
            return BetResult(stake=stake, payout=0.0, profit=-stake, won=False, odds=odds,
                             implied_prob=self._implied_prob(odds))

        margin = int(outcome_str.replace("+", ""))  # winner perspective

        if side == "winner":
            # Winner covers if margin + spread > 0
            won = (margin + spread) > 0
        else:
            # Loser side covers if -(margin + spread) > 0, i.e. margin + spread < 0
            # Equivalently: if loser closes the spread
            won = (margin + spread) < 0

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
    def set_distribution(df: pd.DataFrame) -> pd.Series:
        """
        Compute the empirical distribution of set score outcomes.
        Returns a value_counts Series keyed by "W sets-L sets" (e.g. "2-0").
        """
        valid = df[df["score"].str.contains("RET|W/O", na=True) == False].copy()
        valid = valid.dropna(subset=["winner_sets", "loser_sets"])
        labels = (
            valid["winner_sets"].astype(int).astype(str)
            + "-"
            + valid["loser_sets"].astype(int).astype(str)
        )
        return labels.value_counts(normalize=True).sort_index()
