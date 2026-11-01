"""Shared recommender interface consumed by the evaluator.

The evaluator owns the catalog, user-specific masking and ranking. A model only
needs to assign a score to every requested item for a given user.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Recommender(ABC):
    """Score catalog items for one user.

    ``score`` must return exactly one finite value per item in ``item_ids`` and
    preserve the requested order. Higher scores rank earlier. ``item_ids`` may
    contain any subset of the shared catalog, including items the model did not
    see during training.
    """

    @abstractmethod
    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        """Return one score per item, in the same order as ``item_ids``."""
