from __future__ import annotations

from pathlib import Path

import pytest
from dr_serialize import StrictJsonDecodeError

from whetstone.optimization.gepa import source


def test_source_manifest_round_trips_through_strict_decoder() -> None:
    manifest = source.load_gepa_source_manifest()

    assert manifest["schema"] == source.GEPA_SOURCE_MANIFEST_SCHEMA
    assert (
        manifest["schema_version"]
        == source.GEPA_SOURCE_MANIFEST_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    "raw",
    [b'{"schema":"first","schema":"second"}', b'{"value":NaN}', b"\xff"],
)
def test_source_manifest_rejects_non_strict_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
) -> None:
    (tmp_path / "source_manifest.json").write_bytes(raw)
    monkeypatch.setattr(source.resources, "files", lambda _package: tmp_path)

    with pytest.raises(StrictJsonDecodeError):
        source.load_gepa_source_manifest()
