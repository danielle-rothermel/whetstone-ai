"""Legacy v0 stable ordering helpers for old batch and repair flows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import blake2b

ORDER_KEY_SEPARATOR = "\x1f"
STABLE_ORDER_DIGEST_SIZE = 16


def stable_order_key(*parts: object) -> str:
    encoded: list[str] = []
    for part in parts:
        text = str(part)
        encoded.append(f"{len(text)}{ORDER_KEY_SEPARATOR}{text}")
    return ORDER_KEY_SEPARATOR.join(encoded)


def stable_shuffle[T](
    items: Sequence[T],
    *,
    seed: str,
    key: Callable[[T], str],
) -> list[T]:
    decorated: list[tuple[str, str, int, T]] = []
    for index, item in enumerate(items):
        item_key = key(item)
        decorated.append(
            (
                _stable_digest(seed=seed, item_key=item_key),
                item_key,
                index,
                item,
            )
        )
    return [item for *_order, item in sorted(decorated)]


def _stable_digest(*, seed: str, item_key: str) -> str:
    digest = blake2b(digest_size=STABLE_ORDER_DIGEST_SIZE)
    digest.update(seed.encode())
    digest.update(b"\0")
    digest.update(item_key.encode())
    return digest.hexdigest()
