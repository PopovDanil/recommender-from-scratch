"""Global temporal split shared by every model and experiment.

The two boundaries were chosen by the coverage audit in ``docs/data-audit.md``
and are frozen before tuning. Ratings with equal timestamps stay together
because the assignment depends only on the timestamp value.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import pandas as pd

TRAIN: Final = "train"
VALIDATION: Final = "validation"
TEST: Final = "test"
SPLITS: Final = (TRAIN, VALIDATION, TEST)

POSITIVE: Final = "positive"
EXPLICIT_NEGATIVE: Final = "negative"
NEUTRAL: Final = "neutral"


@dataclass(frozen=True)
class SplitBoundaries:
    """Left-closed validation and test boundaries in UTC."""

    validation_start: datetime
    test_start: datetime

    def __post_init__(self) -> None:
        if self.validation_start.tzinfo is None or self.test_start.tzinfo is None:
            raise ValueError("split boundaries must be timezone-aware")
        if self.test_start <= self.validation_start:
            raise ValueError("test boundary must be after the validation boundary")

    @property
    def validation_epoch(self) -> int:
        return int(self.validation_start.timestamp())

    @property
    def test_epoch(self) -> int:
        return int(self.test_start.timestamp())


DEFAULT_BOUNDARIES: Final = SplitBoundaries(
    validation_start=datetime(2001, 1, 1, tzinfo=UTC),
    test_start=datetime(2001, 7, 1, tzinfo=UTC),
)


def label_ratings(ratings: pd.DataFrame) -> pd.Series:
    """Map ratings to positive (4-5), explicit negative (1-2) or neutral (3)."""
    labels = pd.Series(NEUTRAL, index=ratings.index, dtype="string")
    labels[ratings["rating"] >= 4] = POSITIVE
    labels[ratings["rating"] <= 2] = EXPLICIT_NEGATIVE
    return labels


def assign_splits(
    timestamps: pd.Series,
    boundaries: SplitBoundaries = DEFAULT_BOUNDARIES,
) -> pd.Series:
    """Assign train/validation/test from epoch-second timestamps."""
    validation_epoch = boundaries.validation_epoch
    test_epoch = boundaries.test_epoch

    splits = pd.Series(TEST, index=timestamps.index, dtype="string")
    splits[timestamps < validation_epoch] = TRAIN
    splits[(timestamps >= validation_epoch) & (timestamps < test_epoch)] = VALIDATION
    return splits


def prepare_ratings(
    ratings: pd.DataFrame,
    boundaries: SplitBoundaries = DEFAULT_BOUNDARIES,
) -> pd.DataFrame:
    """Return a copy with ``split`` and ``label`` columns."""
    prepared = ratings.copy()
    prepared["split"] = assign_splits(ratings["timestamp"], boundaries)
    prepared["label"] = label_ratings(ratings)
    return prepared


def build_catalog(train_ratings: pd.DataFrame) -> pd.DataFrame:
    """Movies with at least one positive training rating, plus their counts."""
    positives = train_ratings[train_ratings["label"] == POSITIVE]
    catalog = (
        positives.groupby("movie_id", as_index=False)
        .size()
        .rename(columns={"size": "train_positive_count"})
        .sort_values("movie_id")
        .reset_index(drop=True)
    )
    return catalog
