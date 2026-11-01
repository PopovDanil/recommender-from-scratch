"""Load the frozen split artifacts written by ``scripts/prepare_data.py``."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPLIT_COLUMNS = ("user_id", "movie_id", "rating", "timestamp", "label")
CATALOG_COLUMNS = ("movie_id", "train_positive_count")


@dataclass(frozen=True)
class ProcessedSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    catalog: pd.DataFrame

    @property
    def catalog_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(item) for item in self.catalog["movie_id"].tolist()))


def load_processed_splits(directory: Path) -> ProcessedSplits:
    """Read train/validation/test CSVs and the shared catalog from ``directory``."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} does not exist; run `uv run python -m scripts.prepare_data`"
        )

    def read_split(name: str) -> pd.DataFrame:
        path = directory / f"{name}.csv"
        frame = pd.read_csv(path)
        missing = set(SPLIT_COLUMNS) - set(frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing columns: {names}")
        return frame

    catalog = pd.read_csv(directory / "catalog.csv")
    missing_catalog = set(CATALOG_COLUMNS) - set(catalog.columns)
    if missing_catalog:
        names = ", ".join(sorted(missing_catalog))
        raise ValueError(f"{directory / 'catalog.csv'} is missing columns: {names}")

    return ProcessedSplits(
        train=read_split("train"),
        validation=read_split("validation"),
        test=read_split("test"),
        catalog=catalog,
    )
