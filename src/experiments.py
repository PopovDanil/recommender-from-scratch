"""Experiment grid shared by data preparation and training.

The six configurations are the cartesian product of two models and three target
explicit-negative proportions. Seeds are fixed before any test result is seen.
"""

from dataclasses import dataclass
from typing import Final

MODELS: Final = ("mf_bpr", "lightgcn_bpr")
EXPLICIT_NEGATIVE_PROPORTIONS: Final = (0.0, 0.25, 0.5)
SEEDS: Final = (42, 43, 44)


@dataclass(frozen=True)
class Experiment:
    model: str
    explicit_negative_proportion: float
    seeds: tuple[int, ...] = SEEDS

    @property
    def name(self) -> str:
        percent = round(self.explicit_negative_proportion * 100)
        return f"{self.model}_p{percent}"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model,
            "target_explicit_negative_proportion": self.explicit_negative_proportion,
            "seeds": list(self.seeds),
        }


def experiment_grid() -> tuple[Experiment, ...]:
    return tuple(
        Experiment(model=model, explicit_negative_proportion=proportion)
        for model in MODELS
        for proportion in EXPLICIT_NEGATIVE_PROPORTIONS
    )
