"""Strict user alignment for paired statistics."""


def exact_users(
    left: set[int], right: set[int], *, context: str = "paired comparison"
) -> list[int]:
    missing = sorted(left - right)
    extra = sorted(right - left)
    if missing or extra:
        raise ValueError(f"{context}: missing IDs={missing}; extra IDs={extra}")
    if not left:
        raise ValueError(f"{context}: empty user population")
    return sorted(left)
