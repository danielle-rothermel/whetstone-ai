"""The narrow behavioral-mutant loader for the ed1m env (direct-JSONL read).

ed1m is the behavioral-mutant variant of ed1 (task 18): the encoder describes a
MUTATED HumanEval+ program, the decoder reconstructs it, and the reconstruction
is scored per-input against BOTH the mutant's and the canonical's execution
oracle (:mod:`whetstone.envs.ed1m_oracle`) -- fidelity-to-mutant is the task,
attractor-pull-to-canonical is the REPORTED contamination measurement.

This module reads the behavioral-mutant artifact (``mutants.jsonl`` + optional
``manifest.json``) DIRECTLY, against the stable, documented schema in
``reports/build-behavioral-mutants.md`` -- it deliberately does NOT import
``dr_code.mutants`` (that loader lives on the dr-code ``impl/04`` branch, which
is NOT in the current whetstone checkout). So ed1m builds + tests WITHOUT the
checkout flip; the flip is needed ONLY to REGENERATE the artifact, not to
consume it. This is the single dr-code-mutant surface to re-check on that flip.

Artifact schema (each ``mutants.jsonl`` line; loader-relevant fields):
  * ``task_id`` / ``entry_point`` / ``prompt``
  * ``canonical_full_source`` / ``mutated_full_source`` (full programs)
  * ``operator_family`` / ``seed`` / ``site_description`` / ``diff_summary``
  * ``input_reprs`` -- ``repr([arg1, arg2, ...])`` per test input
  * ``mutant_expected[]`` / ``canonical_expected[]`` -- ``{kind, output_repr}``
    per input, aligned with ``input_reprs``
  * ``distinct_input_indices`` -- the inputs where mutant != canonical (the
    DISCRIMINATING inputs the attractor-pull metric is measured on)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: The canonical behavioral-mutant artifact directory (already generated,
#: config identity d0e082fe...; 204 mutants / 114 tasks).
ED1M_ARTIFACT_DIR = Path(
    "/Users/daniellerothermel/drotherm/data/whetstone-impl/2026-07-22-run1/"
    "mutants/humanevalplus-mutants-v1"
)
ED1M_MUTANTS_FILENAME = "mutants.jsonl"
ED1M_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class ExpectedOutcome:
    """One input's execution-derived expected outcome.

    ``kind`` is ``"value"`` or ``"error"``; ``output_repr`` is ``repr()`` of
    the returned value, or the exception type name for an error outcome. Two
    outcomes compare equal iff BOTH kind and output_repr match -- the oracle
    compares a reconstruction's outcome to the recorded outcome by this
    equality (never by a looser value coercion).
    """

    kind: str
    output_repr: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ExpectedOutcome:
        return cls(
            kind=str(data["kind"]),
            output_repr=str(data["output_repr"]),
        )


@dataclass(frozen=True, slots=True)
class Ed1mMutant:
    """One accepted behavioral mutant, loader-ready for the ed1m env.

    ``mutated_full_source`` is the encoder's INPUT_CODE (the buggy program);
    ``mutant_expected`` / ``canonical_expected`` (aligned with ``input_reprs``)
    are the dual oracle the reconstruction is scored against;
    ``distinct_input_indices`` are the discriminating inputs (mutant !=
    canonical) the attractor-pull is measured on.
    """

    task_id: str
    entry_point: str
    prompt: str
    canonical_full_source: str
    mutated_full_source: str
    operator_family: str
    seed: int
    site_description: str
    diff_summary: str
    input_reprs: tuple[str, ...]
    mutant_expected: tuple[ExpectedOutcome, ...]
    canonical_expected: tuple[ExpectedOutcome, ...]
    distinct_input_indices: tuple[int, ...]
    optional_identifier_rename: str | None = None

    #: A stable per-mutant id (task + family + seed + site) -- the ed1m
    #: Instance id, distinct across a task's several mutants.
    @property
    def mutant_id(self) -> str:
        return (
            f"{self.task_id}::{self.operator_family}"
            f"::s{self.seed}::n{self.site_description_tag}"
        )

    @property
    def site_description_tag(self) -> str:
        """A filesystem/id-safe tag of the site description."""
        return "".join(
            c if c.isalnum() else "_" for c in self.site_description
        ).strip("_")

    @property
    def distinct_input_count(self) -> int:
        return len(self.distinct_input_indices)

    @classmethod
    def from_record(cls, data: dict[str, object]) -> Ed1mMutant:
        def _expected(key: str) -> tuple[ExpectedOutcome, ...]:
            raw = data[key]
            assert isinstance(raw, list)
            out: list[ExpectedOutcome] = []
            for e in raw:
                assert isinstance(e, dict)
                out.append(
                    ExpectedOutcome(
                        kind=str(e.get("kind")),
                        output_repr=str(e.get("output_repr")),
                    )
                )
            return tuple(out)

        distinct = data["distinct_input_indices"]
        assert isinstance(distinct, list)
        inputs = data["input_reprs"]
        assert isinstance(inputs, list)
        seed = data["seed"]
        assert isinstance(seed, int)
        rename = data.get("optional_identifier_rename")
        return cls(
            task_id=str(data["task_id"]),
            entry_point=str(data["entry_point"]),
            prompt=str(data["prompt"]),
            canonical_full_source=str(data["canonical_full_source"]),
            mutated_full_source=str(data["mutated_full_source"]),
            operator_family=str(data["operator_family"]),
            seed=seed,
            site_description=str(data["site_description"]),
            diff_summary=str(data["diff_summary"]),
            input_reprs=tuple(str(x) for x in inputs),
            mutant_expected=_expected("mutant_expected"),
            canonical_expected=_expected("canonical_expected"),
            distinct_input_indices=tuple(
                i for i in distinct if isinstance(i, int)
            ),
            optional_identifier_rename=(
                str(rename) if rename is not None else None
            ),
        )


def load_ed1m_mutants(
    path: Path | None = None, *, limit: int | None = None
) -> tuple[Ed1mMutant, ...]:
    """Load behavioral mutants from ``mutants.jsonl`` (direct read).

    ``path`` defaults to the canonical ``mutants.jsonl``. Records are
    read in file order (the artifact is deterministically sorted by
    ``(task_id, family, seed)``), so a ``limit`` yields a stable first-N slice.
    """
    jsonl = path or (ED1M_ARTIFACT_DIR / ED1M_MUTANTS_FILENAME)
    mutants: list[Ed1mMutant] = []
    with jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            mutants.append(Ed1mMutant.from_record(json.loads(line)))
            if limit is not None and len(mutants) >= limit:
                break
    return tuple(mutants)


def ed1m_manifest_identity(path: Path | None = None) -> str | None:
    """The mutant-suite config identity from the manifest (pinned provenance).

    Read for the ed1m dataset-revision / provenance record so a cell line pins
    the exact mutant suite it ran against. ``None`` when no manifest.
    """
    manifest = path or (ED1M_ARTIFACT_DIR / ED1M_MANIFEST_FILENAME)
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text())
    identity = data.get("config_identity")
    return str(identity) if identity is not None else None


__all__ = [
    "ED1M_ARTIFACT_DIR",
    "ED1M_MANIFEST_FILENAME",
    "ED1M_MUTANTS_FILENAME",
    "Ed1mMutant",
    "ExpectedOutcome",
    "ed1m_manifest_identity",
    "load_ed1m_mutants",
]
