from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from whetstone.optim.gepa.factory import CanonicalGepaAdapterFactory


def _factory() -> tuple[CanonicalGepaAdapterFactory, MagicMock]:
    control = MagicMock()
    control.identity_hash.return_value = "a" * 64
    control.prompt_binding_identity_hash = "b" * 64
    control.prompt_format_identity_hash = "c" * 64
    control.component_names = ("generate",)
    control.gepa_source_manifest_hash = "d" * 64

    evaluation_authority = MagicMock()
    evaluation_authority.control_identity_hash = "a" * 64
    evaluation_authority.component_names = ("generate",)
    evaluation_authority.runtime_hash = "e" * 64

    proposal_authority = MagicMock()
    proposal_authority.control_identity_hash = "a" * 64
    proposal_authority.runtime_hash = "f" * 64

    prompt_services = MagicMock()
    prompt_services.binding.identity_hash.return_value = "b" * 64
    prompt_services.descriptor.identity_hash.return_value = "c" * 64
    component = MagicMock()
    component.component_name = "generate"
    prompt_services.descriptor.components = (component,)

    factory = CanonicalGepaAdapterFactory(
        store=MagicMock(),
        run_id="run-1",
        control=control,
        evaluation_authority=evaluation_authority,
        proposal_authority=proposal_authority,
        prompt_services=prompt_services,
    )
    return factory, control


def test_harness_create_does_not_import_effect_runtime() -> None:
    factory, control = _factory()
    with patch.dict(sys.modules, {"whetstone.optim.gepa.effect_runtime": None}):
        with patch("whetstone.optim.gepa.factory.WhetstoneGepaAdapter") as adapter_cls:
            adapter = factory.create(control=control, effect_broker="harness")
    adapter_cls.assert_called_once()
    assert adapter is adapter_cls.return_value


def test_dbos_create_imports_effect_runtime() -> None:
    factory, control = _factory()
    with patch.dict(sys.modules, {"whetstone.optim.gepa.effect_runtime": None}):
        with pytest.raises(ImportError):
            factory.create(control=control, effect_broker="dbos")
