from __future__ import annotations

import json

import pytest

from whetstone.platform.integrity import (
    INTEGRITY_KEY_ID_ENV,
    INTEGRITY_PRIVATE_KEY_PATH_ENV,
    INTEGRITY_PUBLIC_KEY_RING_ENV,
    required_bundle_integrity_configuration,
)


def test_integrity_configuration_requires_every_operator_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(INTEGRITY_KEY_ID_ENV, raising=False)
    monkeypatch.delenv(INTEGRITY_PRIVATE_KEY_PATH_ENV, raising=False)
    monkeypatch.delenv(INTEGRITY_PUBLIC_KEY_RING_ENV, raising=False)

    with pytest.raises(ValueError, match=INTEGRITY_KEY_ID_ENV):
        required_bundle_integrity_configuration()


def test_integrity_configuration_rejects_key_not_in_public_ring(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    private_key = tmp_path / "private.pem"
    private_key.touch()
    monkeypatch.setenv(INTEGRITY_KEY_ID_ENV, "current")
    monkeypatch.setenv(INTEGRITY_PRIVATE_KEY_PATH_ENV, str(private_key))
    monkeypatch.setenv(
        INTEGRITY_PUBLIC_KEY_RING_ENV, json.dumps({"previous": "public-key"})
    )

    with pytest.raises(ValueError, match="must include"):
        required_bundle_integrity_configuration()
