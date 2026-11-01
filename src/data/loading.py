"""Load the fixed MovieLens-1M release.

Raw files use ``::`` separators and epoch-second timestamps. The loader keeps
the integer identifiers from the release; remapping belongs to model code.
"""

from pathlib import Path

import pandas as pd

RATING_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")
USER_COLUMNS = ("user_id", "gender", "age", "occupation", "zip_code")
MOVIE_COLUMNS = ("movie_id", "title", "genres")


def load_ratings(path: Path) -> pd.DataFrame:
    """Load ``ratings.dat`` with columns user_id, movie_id, rating, timestamp."""
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=RATING_COLUMNS,
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
    )
    _validate_ratings(ratings)
    return ratings


def load_users(path: Path) -> pd.DataFrame:
    """Load ``users.dat`` with raw demographic columns."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=USER_COLUMNS,
        dtype={
            "user_id": "int64",
            "gender": "string",
            "age": "int64",
            "occupation": "int64",
            "zip_code": "string",
        },
    )


def load_movies(path: Path) -> pd.DataFrame:
    """Load ``movies.dat`` with raw titles and pipe-separated genres."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=MOVIE_COLUMNS,
        dtype={"movie_id": "int64", "title": "string", "genres": "string"},
    )


def _validate_ratings(ratings: pd.DataFrame) -> None:
    if ratings.isna().any().any():
        raise ValueError("ratings.dat contains missing values")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("ratings must be in the range 1-5")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("ratings.dat contains duplicate user-movie pairs")
