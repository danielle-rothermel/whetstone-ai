from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from importlib import resources
from pathlib import Path
from typing import Any

from whetstone.core.identity import compute_identity_hash

GEPA_DISTRIBUTION_NAME = "gepa"
GEPA_DISTRIBUTION_VERSION = "0.1.1"
GEPA_REPOSITORY_TAG = "v0.1.1"
GEPA_REPOSITORY_COMMIT = "b4dbb55b7601dac448cdb836d5a401ca7d9eb920"
GEPA_WHEEL_SHA256 = (
    "71ead7c591eafcc727b83509cdc4182f20264800a6ddf8520d61419daeb47466"
)
GEPA_SDIST_SHA256 = (
    "643fda01c23de4c9f01306e01305dd69facc29bcb34ad59e4cd07e6621d34aa1"
)
GEPA_LICENSE_SHA256 = (
    "10c47467a961feb40adf3294fe27dd9cba79d4d1b7cf27173b1c34586d4126c3"
)
GEPA_SOURCE_MANIFEST_SCHEMA = "whetstone.gepa.upstream_source_manifest"
GEPA_SOURCE_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_RESOURCE = "source_manifest.json"


class GepaSourceMismatchError(RuntimeError):
    """The installed GEPA distribution does not match the frozen source."""


def load_gepa_source_manifest() -> dict[str, Any]:
    """Load and minimally validate the committed source manifest."""

    resource = resources.files("whetstone.optimization.gepa").joinpath(
        _MANIFEST_RESOURCE
    )
    manifest = json.loads(resource.read_text(encoding="utf-8"))
    if manifest.get("schema") != GEPA_SOURCE_MANIFEST_SCHEMA:
        raise GepaSourceMismatchError("GEPA source manifest schema drift")
    if manifest.get("schema_version") != GEPA_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise GepaSourceMismatchError(
            "GEPA source manifest schema-version drift"
        )
    distribution = manifest.get("distribution")
    repository = manifest.get("repository")
    expected_distribution = {
        "name": GEPA_DISTRIBUTION_NAME,
        "version": GEPA_DISTRIBUTION_VERSION,
        "wheel_sha256": GEPA_WHEEL_SHA256,
        "sdist_sha256": GEPA_SDIST_SHA256,
    }
    expected_repository = {
        "tag": GEPA_REPOSITORY_TAG,
        "commit": GEPA_REPOSITORY_COMMIT,
    }
    if distribution != expected_distribution:
        raise GepaSourceMismatchError(
            "GEPA source manifest distribution identity drift"
        )
    if repository != expected_repository:
        raise GepaSourceMismatchError(
            "GEPA source manifest repository identity drift"
        )
    if manifest.get("license") != {
        "expression": "MIT",
        "file": "gepa-0.1.1.dist-info/licenses/LICENSE",
        "sha256": GEPA_LICENSE_SHA256,
    }:
        raise GepaSourceMismatchError("GEPA source manifest license drift")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise GepaSourceMismatchError(
            "GEPA source manifest must bind source files"
        )
    return manifest


def gepa_source_manifest_hash() -> str:
    """Return the canonical identity hash bound into every GEPA run."""

    manifest = load_gepa_source_manifest()
    return compute_identity_hash(
        schema=GEPA_SOURCE_MANIFEST_SCHEMA,
        schema_version=GEPA_SOURCE_MANIFEST_SCHEMA_VERSION,
        payload={
            key: value
            for key, value in manifest.items()
            if key not in {"schema", "schema_version"}
        },
    )


GEPA_SOURCE_MANIFEST_HASH = gepa_source_manifest_hash()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_installed_gepa_source() -> str:
    """Fail closed unless the importable GEPA package is the frozen build.

    The artifact hashes in the manifest bind dependency resolution. This guard
    additionally verifies the installed version, import origin, and every
    algorithm source file used transitively by ``gepa.optimize``.
    """

    manifest = load_gepa_source_manifest()
    try:
        distribution = importlib.metadata.distribution(GEPA_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError as exc:
        raise GepaSourceMismatchError(
            "frozen gepa==0.1.1 is not installed"
        ) from exc
    if distribution.version != GEPA_DISTRIBUTION_VERSION:
        raise GepaSourceMismatchError(
            "installed GEPA version drift: "
            f"expected {GEPA_DISTRIBUTION_VERSION}, "
            f"got {distribution.version}"
        )

    spec = importlib.util.find_spec("gepa")
    if spec is None or spec.origin is None:
        raise GepaSourceMismatchError("the GEPA package is not importable")
    expected_origin = Path(
        str(distribution.locate_file("gepa/__init__.py"))
    ).resolve()
    if Path(spec.origin).resolve() != expected_origin:
        raise GepaSourceMismatchError(
            "importable GEPA source is not owned by the frozen distribution"
        )

    mismatches: list[str] = []
    license_record = manifest["license"]
    license_path = Path(str(distribution.locate_file(license_record["file"])))
    if (
        not license_path.is_file()
        or _sha256(license_path) != license_record["sha256"]
    ):
        mismatches.append(f"{license_record['file']}: license drift")
    for relative_path, expected_hash in manifest["source_files"].items():
        installed_path = Path(str(distribution.locate_file(relative_path)))
        if not installed_path.is_file():
            mismatches.append(f"{relative_path}: missing")
            continue
        observed_hash = _sha256(installed_path)
        if observed_hash != expected_hash:
            mismatches.append(
                f"{relative_path}: expected {expected_hash}, "
                f"got {observed_hash}"
            )
    if mismatches:
        raise GepaSourceMismatchError(
            "installed GEPA source drift:\n" + "\n".join(mismatches)
        )
    return GEPA_SOURCE_MANIFEST_HASH


__all__ = [
    "GEPA_DISTRIBUTION_NAME",
    "GEPA_DISTRIBUTION_VERSION",
    "GEPA_LICENSE_SHA256",
    "GEPA_REPOSITORY_COMMIT",
    "GEPA_REPOSITORY_TAG",
    "GEPA_SDIST_SHA256",
    "GEPA_SOURCE_MANIFEST_HASH",
    "GEPA_SOURCE_MANIFEST_SCHEMA",
    "GEPA_SOURCE_MANIFEST_SCHEMA_VERSION",
    "GEPA_WHEEL_SHA256",
    "GepaSourceMismatchError",
    "gepa_source_manifest_hash",
    "load_gepa_source_manifest",
    "verify_installed_gepa_source",
]
