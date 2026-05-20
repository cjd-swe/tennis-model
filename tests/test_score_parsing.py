"""Unit tests for Sackmann score parsing."""

import pytest
from tennis_model.ingest.sackmann import (
    _parse_sets,
    _score_to_total_games,
    _count_winner_sets,
    _count_loser_sets,
)


@pytest.mark.parametrize("score, expected_total", [
    ("6-4 6-2", 18),           # (6+4)+(6+2) = 18
    ("6-4 3-6 6-3", 28),       # (6+4)+(3+6)+(6+3) = 28
    ("7-6(3) 6-4", 23),        # (7+6)+(6+4) = 23
    ("6-4 6-4 6-2", 28),       # (6+4)+(6+4)+(6+2) = 28
    ("4-6 6-3 6-4 3-6 6-3", 47),  # (4+6)+(6+3)+(6+4)+(3+6)+(6+3) = 47
    # Retirement: parser strips " RET" and totals completed sets — market layer
    # is responsible for filtering retirements, not the raw parser.
    ("6-4 6-4 RET", 20),
    ("W/O", None),            # walkover → None (no games parsed)
    (None, None),
])
def test_score_to_total_games(score, expected_total):
    assert _score_to_total_games(score) == expected_total


@pytest.mark.parametrize("score, w_sets, l_sets", [
    ("6-4 6-2", 2, 0),
    ("6-4 3-6 6-3", 2, 1),
    ("4-6 6-3 7-6(2)", 2, 1),
    ("4-6 6-3 6-4 3-6 6-3", 3, 2),
])
def test_set_counts(score, w_sets, l_sets):
    assert _count_winner_sets(score) == w_sets
    assert _count_loser_sets(score) == l_sets
