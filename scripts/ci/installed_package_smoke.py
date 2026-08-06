from __future__ import annotations

import importlib
import pkgutil
import sys
from importlib import resources
from pathlib import Path
from types import ModuleType

import whetstone
from whetstone.coordination.evaluation_claims import EvaluationClaims
from whetstone.core.identity import TypedRef
from whetstone.envs.factory import EnvExperiment
from whetstone.evaluation.schema import EvaluationEvidence
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.contracts import OptimizationRun
from whetstone.optimization.gepa.source import load_gepa_source_manifest
from whetstone.provider.attempt import ProviderCallResult

EXPECTED_PACKAGE_ROOTS = {
    "whetstone.coordination",
    "whetstone.core",
    "whetstone.envs",
    "whetstone.evaluation",
    "whetstone.execution",
    "whetstone.experiment",
    "whetstone.optimization",
    "whetstone.provider",
    "whetstone.runner",
}
REPRESENTATIVE_TYPES = (
    EvaluationClaims,
    TypedRef,
    EnvExperiment,
    EvaluationEvidence,
    PromptResultCache,
    Candidate,
    OptimizationRun,
    ProviderCallResult,
)


def _module_origin(module: ModuleType) -> Path:
    origin = module.__file__
    if origin is None:
        raise AssertionError(f"{module.__name__} has no filesystem origin")
    return Path(origin).resolve()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: installed_package_smoke.py CHECKOUT_ROOT")
    checkout_root = Path(sys.argv[1]).resolve(strict=True)

    package_roots = {
        discovered.name
        for discovered in pkgutil.iter_modules(
            whetstone.__path__, prefix="whetstone."
        )
        if discovered.ispkg
    }
    if package_roots != EXPECTED_PACKAGE_ROOTS:
        raise AssertionError(
            "installed package roots differ: "
            f"expected {sorted(EXPECTED_PACKAGE_ROOTS)}, "
            f"got {sorted(package_roots)}"
        )

    imported_modules = {
        whetstone,
        *(importlib.import_module(name) for name in EXPECTED_PACKAGE_ROOTS),
        *(
            importlib.import_module(representative.__module__)
            for representative in REPRESENTATIVE_TYPES
        ),
    }
    for module in imported_modules:
        origin = _module_origin(module)
        if origin.is_relative_to(checkout_root):
            raise AssertionError(
                f"{module.__name__} resolved inside the checkout: {origin}"
            )

    if not resources.files("whetstone").joinpath("py.typed").is_file():
        raise AssertionError("installed wheel is missing whetstone/py.typed")

    manifest = load_gepa_source_manifest()
    if manifest.get("schema") != "whetstone.gepa.upstream_source_manifest":
        raise AssertionError("installed GEPA manifest has the wrong schema")
    if manifest.get("schema_version") != 1:
        raise AssertionError(
            "installed GEPA manifest has the wrong schema version"
        )


if __name__ == "__main__":
    main()
