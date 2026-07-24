"""Collision-free Provider Quota Identity label derivation.

Provider Concurrency Control uses dr-platform's generic exact-label
selector-capacity admission as a deliberately coarse, conservative
cross-worker cap. The concurrency *unit* is one dr-providers Provider Quota
Identity, exactly ``(provider, protocol, model)`` — no credential, account,
or override component. dr-platform enforces label/selector matching
generically; **Whetstone alone** derives the labels and owns their meaning.

Two Whetstone-owned properties are load-bearing:

* **Collision-freedom.** Two Provider Quota Identities derive the same label
  if and only if they are equal. dr-providers' own :meth:`label` joins the
  three fields with ``":"``, which is *not* injective: a ``model`` value may
  itself contain ``":"`` (e.g. ``"meta-llama/llama-3:70b"``), so two distinct
  tuples could collide. This module derives a length-delimited value that no
  delimiter-containing name can forge, so a per-label Stage capacity always
  caps exactly its intended quota.

* **A single fixed label key.** Every derived label shares one reserved label
  *key* (:data:`QUOTA_LABEL_KEY`); the injective encoding lives entirely in
  the label *value*. This keeps a Work Item's label map a flat
  ``{key: value}`` shape dr-platform matches with an exact-label selector,
  and lets multi-route work carry one entry per applicable quota by using a
  per-quota key suffix (:func:`quota_label`).

Work that may route through more than one ``(provider, protocol, model)``
tuple (an encoder and a decoder, or a fallback route) carries **every**
applicable label so every matching Stage Control applies together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr_providers import ProviderQuotaIdentity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "QUOTA_LABEL_KEY",
    "QUOTA_LABEL_VALUE_VERSION",
    "quota_label",
    "quota_label_value",
    "quota_labels_for",
    "quota_selector",
]

#: The reserved label key every derived Provider Quota label shares. A single
#: fixed key keeps the label map a flat exact-match shape for dr-platform's
#: selector admission. Multi-route work distinguishes its several quota labels
#: by a per-quota key suffix (see :func:`quota_label`).
QUOTA_LABEL_KEY = "whetstone.provider_quota"

#: Version tag on the value encoding so the derived label string is
#: self-describing and can never be confused with a differently versioned or
#: differently owned encoding.
QUOTA_LABEL_VALUE_VERSION = "v1"

# A separator that cannot appear inside a length-prefixed field: each field is
# emitted as ``<decimal-length>:<bytes>`` so the value is self-delimiting and
# parsing never depends on the field contents.
_FIELD_SEP = ":"


def _length_prefixed(text: str) -> str:
    """Encode one field as ``<len>:<text>`` (self-delimiting, injective)."""
    return f"{len(text)}{_FIELD_SEP}{text}"


def quota_label_value(quota: ProviderQuotaIdentity) -> str:
    """Derive the collision-free label *value* for one Quota Identity.

    The value is a version tag followed by the three tuple fields, each
    length-prefixed as ``<len>:<field>``. Length prefixing makes the encoding
    injective over *arbitrary* field contents — including a ``model`` name that
    itself contains ``":"``, ``"/"``, or the version tag — so two distinct
    ``(provider, protocol, model)`` tuples can never derive the same value.
    """
    if not isinstance(quota, ProviderQuotaIdentity):
        raise TypeError("quota must be a ProviderQuotaIdentity")
    fields = (quota.provider.value, quota.protocol.value, quota.model)
    return QUOTA_LABEL_VALUE_VERSION + "".join(
        _length_prefixed(field) for field in fields
    )


def quota_label(quota: ProviderQuotaIdentity) -> tuple[str, str]:
    """Derive the ``(key, value)`` label entry for one Quota Identity.

    The key is :data:`QUOTA_LABEL_KEY` suffixed with the collision-free value,
    so a Work Item carrying several routes' labels holds one distinct entry per
    quota (a flat map cannot hold two values for one key). The value repeats
    the collision-free encoding so an exact-label selector matches on it
    directly.
    """
    value = quota_label_value(quota)
    return f"{QUOTA_LABEL_KEY}/{value}", value


def quota_labels_for(
    quotas: Iterable[ProviderQuotaIdentity],
) -> dict[str, str]:
    """Derive the complete label map for work routing through ``quotas``.

    Multi-route work (more than one applicable ``(provider, protocol, model)``
    tuple) carries **every** applicable label so every matching Stage Control
    applies together. Duplicate quotas collapse to one entry; the result is
    order-independent (a plain mapping keyed by the collision-free key).
    """
    labels: dict[str, str] = {}
    for quota in quotas:
        key, value = quota_label(quota)
        labels[key] = value
    return labels


def quota_selector(quota: ProviderQuotaIdentity) -> Mapping[str, str]:
    """The exact-label selector a per-quota Stage capacity is keyed by.

    A control configured with this selector admits exactly the work whose
    label map contains this quota's entry — the same ``(key, value)`` pair
    :func:`quota_label` puts on the Work Item — and nothing else, because the
    value is collision-free.
    """
    key, value = quota_label(quota)
    return {key: value}
