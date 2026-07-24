"""Collision-free Provider Quota Identity label derivation.

Proves the Whetstone-owned label derivation is injective over arbitrary
``(provider, protocol, model)`` tuples — including delimiter-containing model
names that would collide under a naive ``":"`` join — and that multi-route
work carries every applicable label.
"""

from __future__ import annotations

from itertools import combinations

from dr_providers import Protocol, ProviderKind, ProviderQuotaIdentity

from whetstone.orchestration.labels import (
    QUOTA_LABEL_KEY,
    quota_label,
    quota_label_value,
    quota_labels_for,
    quota_selector,
)


def _quota(*, model: str) -> ProviderQuotaIdentity:
    return ProviderQuotaIdentity(
        provider=ProviderKind.OPENROUTER,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )


def test_derivation_is_deterministic() -> None:
    quota = _quota(model="gpt-4o")
    assert quota_label_value(quota) == quota_label_value(quota)
    assert quota_label(quota) == quota_label(quota)


def test_equal_tuples_derive_equal_labels() -> None:
    a = _quota(model="gpt-4o")
    b = _quota(model="gpt-4o")
    assert quota_label(a) == quota_label(b)


def test_distinct_models_derive_distinct_labels() -> None:
    a = _quota(model="gpt-4o")
    b = _quota(model="gpt-4o-mini")
    assert quota_label(a) != quota_label(b)


def test_delimiter_containing_names_stay_distinct() -> None:
    """Collision-freedom over delimiter-containing model names.

    The derivation is injective even when the model contains the label
    delimiters (``:``, ``/``) and the version tag. Length prefixing makes the
    encoding self-delimiting, so no delimiter-containing name can forge the
    encoding of a different tuple — a robustness dr-providers' plain ``":"``
    join does not itself carry.
    """
    a = _quota(model="chat_completions:llama")
    b = _quota(model="llama")
    c = _quota(model="chat_completions/llama")
    assert quota_label(a) != quota_label(b)
    assert quota_label(a) != quota_label(c)
    assert quota_label_value(a) != quota_label_value(b)
    # The value is self-delimiting: the field lengths are recoverable, so the
    # three fields cannot re-associate into a different tuple.
    assert quota_label_value(a).startswith("v1")


def test_collision_freedom_over_a_delimiter_heavy_family() -> None:
    """Exhaustive pairwise distinctness over adversarial model names.

    Every name is built from the label delimiters (``:``, ``/``, the version
    tag ``v1``, digits) so a non-injective encoding would collapse some pair.
    Length prefixing guarantees all are distinct.
    """
    models = [
        "a",
        "a:b",
        "a:b:c",
        "1:a",
        "11:a",
        "v1",
        "v110:openrouter",
        "openrouter:chat_completions:a",
        "a/b",
        ":",
        "::",
    ]
    quotas = [_quota(model=model) for model in models]
    values = [quota_label_value(q) for q in quotas]
    assert len(set(values)) == len(values)
    for left, right in combinations(quotas, 2):
        assert quota_label(left) != quota_label(right)


def test_provider_and_protocol_participate_in_the_label() -> None:
    chat = ProviderQuotaIdentity(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model="gpt-4o",
    )
    responses = ProviderQuotaIdentity(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.RESPONSES,
        model="gpt-4o",
    )
    anthropic = ProviderQuotaIdentity(
        provider=ProviderKind.ANTHROPIC,
        protocol=Protocol.CHAT_COMPLETIONS,
        model="gpt-4o",
    )
    labels = {
        quota_label(chat),
        quota_label(responses),
        quota_label(anthropic),
    }
    assert len(labels) == 3


def test_label_key_is_the_reserved_prefix() -> None:
    key, value = quota_label(_quota(model="gpt-4o"))
    assert key.startswith(QUOTA_LABEL_KEY)
    assert value in key


def test_selector_matches_its_own_label_entry() -> None:
    quota = _quota(model="gpt-4o")
    key, value = quota_label(quota)
    selector = quota_selector(quota)
    assert dict(selector) == {key: value}


def test_multi_route_work_carries_every_applicable_label() -> None:
    encoder = _quota(model="encoder-model")
    decoder = _quota(model="decoder-model")
    labels = quota_labels_for([encoder, decoder])
    assert len(labels) == 2
    # Each route's selector is contained by the multi-route label map, so every
    # matching Stage Control applies to this work item.
    for route in (encoder, decoder):
        selector = quota_selector(route)
        assert all(labels.get(k) == v for k, v in selector.items())


def test_duplicate_quotas_collapse_to_one_entry() -> None:
    quota = _quota(model="gpt-4o")
    labels = quota_labels_for([quota, quota, quota])
    assert len(labels) == 1


def test_empty_quota_set_yields_empty_labels() -> None:
    assert quota_labels_for([]) == {}
