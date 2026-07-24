"""Tests for the task-selection manifest reader + role-true splits (task 29).

The runner consumes a run's ``whetstone.run.task_selection/v1`` manifest with
TRUE train/val/test semantics: internal = train+val (membership), official =
test EXACTLY (membership, NOT a first-N slice), with the manifest's content
hash + pool folded into each split's ``eval_config_hash``. These tests drive
the loader, the env builders' manifest path, and the CLI refusals -- no live
call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment, load_ed1_tasks
from whetstone.runner.cli import _build_cell_config
from whetstone.runner.task_split_manifest import (
    TASK_SELECTION_SCHEMA,
    TaskSplitManifestError,
    load_task_split_manifest,
    resolve_manifest_split,
)


def _cell_args(env: str, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        optimizer="eval",
        env=env,
        lane="openrouter",
        attempt=0,
        task_model=None,
        proposer_model=None,
        proposer_cli=None,
        non_canonical=False,
        execution_mode="in-process",
        concurrency=4,
        max_wall_seconds=3600.0,
        official_n=None,
        official_repeats=None,
        missing_data=None,
        max_skip_fraction=None,
        dry_run_fake=False,
        live=True,
        task_filter=None,
        task_split_manifest=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _pool(limit: int = 12):
    return load_ed1_tasks(prefer_snapshot=True, limit=limit)


def _ids(split) -> tuple[str, ...]:
    return tuple(str(i.id) for i in split.instances)


def _manifest_dict(
    *,
    ed1_train=("HumanEval/0", "HumanEval/1"),
    ed1_val=("HumanEval/2",),
    ed1_test=("HumanEval/3", "HumanEval/4"),
    d1_train=("HumanEval/5",),
    d1_val=("HumanEval/6",),
    d1_test=("HumanEval/7", "HumanEval/8"),
    schema=TASK_SELECTION_SCHEMA,
) -> dict:
    return {
        "schema": schema,
        "seed": 1,
        "pools": {
            "ed1": {
                "arm": "encdec_naive",
                "dropped_always_pass": [],
                "train": list(ed1_train),
                "val": list(ed1_val),
                "test": list(ed1_test),
            },
            "d1": {
                "arm": "direct_original",
                "dropped_always_pass": [],
                "train": list(d1_train),
                "val": list(d1_val),
                "test": list(d1_test),
            },
        },
    }


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "manifest-v1.json"
    p.write_text(json.dumps(data))
    return p


# --- Loader validation -------------------------------------------------------


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    p = _write(tmp_path, _manifest_dict(schema="something/else"))
    with pytest.raises(TaskSplitManifestError, match="schema"):
        load_task_split_manifest(p)


def test_load_rejects_missing_pools(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema": TASK_SELECTION_SCHEMA}))
    with pytest.raises(TaskSplitManifestError, match="pools"):
        load_task_split_manifest(p)


def test_load_rejects_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text("{not json")
    with pytest.raises(TaskSplitManifestError, match="valid JSON"):
        load_task_split_manifest(p)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TaskSplitManifestError, match="not found"):
        load_task_split_manifest(tmp_path / "nope.json")


def test_content_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    b = load_task_split_manifest(
        _write(tmp_path, _manifest_dict())
    )
    assert a.content_hash == b.content_hash
    c = load_task_split_manifest(
        _write(tmp_path, _manifest_dict(ed1_test=("HumanEval/3",)))
    )
    assert c.content_hash != a.content_hash


# --- Role mapping ------------------------------------------------------------


def test_internal_is_train_plus_val_official_is_test(tmp_path: Path) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    roles = m.for_env("ed1")
    assert roles.pool_key == "ed1"
    # internal = train FOLLOWED BY val (order preserved, identity-bearing).
    assert roles.internal_ids == ("HumanEval/0", "HumanEval/1", "HumanEval/2")
    assert roles.official_ids == ("HumanEval/3", "HumanEval/4")


def test_d1_selects_d1_pool(tmp_path: Path) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    roles = m.for_env("d1")
    assert roles.pool_key == "d1"
    assert roles.internal_ids == ("HumanEval/5", "HumanEval/6")
    assert roles.official_ids == ("HumanEval/7", "HumanEval/8")


def test_duplicate_role_ids_refused(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        _manifest_dict(ed1_train=("HumanEval/0", "HumanEval/0")),
    )
    m = load_task_split_manifest(p)
    with pytest.raises(TaskSplitManifestError, match="duplicate"):
        m.for_env("ed1")


# --- ed1m + unknown-env refusals --------------------------------------------


def test_ed1m_is_refused(tmp_path: Path) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    with pytest.raises(TaskSplitManifestError, match="ed1m"):
        m.for_env("ed1m")


def test_unknown_env_is_refused(tmp_path: Path) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    with pytest.raises(TaskSplitManifestError, match="applies only"):
        m.for_env("c18")


# --- Membership splits through the env builders ------------------------------


@pytest.mark.parametrize("env", ["ed1", "d1"])
def test_builder_membership_split(tmp_path: Path, env: str) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    roles = m.for_env(env)
    pool = _pool(12)
    build = build_ed1_experiment if env == "ed1" else build_d1_experiment
    exp = build(tasks=pool, split_manifest=roles)
    internal_ids = _ids(exp.eval_configs.internal)
    official_ids = _ids(exp.eval_configs.official)
    assert set(internal_ids) == set(roles.internal_ids)
    assert set(official_ids) == set(roles.official_ids)
    # disjoint: official is test-only, never overlapping internal.
    assert not (set(internal_ids) & set(official_ids))


def test_unknown_manifest_id_names_offenders(tmp_path: Path) -> None:
    # A manifest test id absent from the loaded pool is a typed refusal naming
    # the offender -- never a silent drop.
    m = load_task_split_manifest(
        _write(tmp_path, _manifest_dict(ed1_test=("HumanEval/9999",)))
    )
    roles = m.for_env("ed1")
    with pytest.raises(TaskSplitManifestError, match="HumanEval/9999"):
        resolve_manifest_split(
            roles=roles,
            items=_pool(4),
            id_of=lambda t: str(t.instance.id),
        )


# --- --official-n caps within the test set -----------------------------------


def test_official_n_caps_within_test(tmp_path: Path) -> None:
    m = load_task_split_manifest(
        _write(
            tmp_path,
            _manifest_dict(
                ed1_test=("HumanEval/3", "HumanEval/4", "HumanEval/5")
            ),
        )
    )
    roles = m.for_env("ed1")
    exp = build_ed1_experiment(
        tasks=_pool(12), split_manifest=roles, official_n=2
    )
    # first-2 of the test set, in manifest order.
    assert _ids(exp.eval_configs.official) == ("HumanEval/3", "HumanEval/4")


def test_official_n_over_test_size_uses_all(tmp_path: Path) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    roles = m.for_env("ed1")
    # test has 2 ids; a cap of 99 is a loud note, not an error, using all 2.
    exp = build_ed1_experiment(
        tasks=_pool(12), split_manifest=roles, official_n=99
    )
    assert set(_ids(exp.eval_configs.official)) == set(roles.official_ids)


# --- Identity folding --------------------------------------------------------


def test_manifest_folds_into_eval_config_hash(tmp_path: Path) -> None:
    m = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    roles = m.for_env("ed1")
    pool = _pool(12)
    manifest_exp = build_ed1_experiment(tasks=pool, split_manifest=roles)
    # A non-manifest cell over the SAME first-N pool has a DISTINCT identity
    # (the manifest tag folds in), proving the fold is real.
    plain_exp = build_ed1_experiment(tasks=pool)
    m_hash = (
        manifest_exp.eval_configs.official.eval_config.config_identity_hash
    )
    p_hash = plain_exp.eval_configs.official.eval_config.config_identity_hash
    assert m_hash != p_hash


def test_distinct_manifests_distinct_identity(tmp_path: Path) -> None:
    pool = _pool(12)
    a = load_task_split_manifest(_write(tmp_path, _manifest_dict()))
    b = load_task_split_manifest(
        _write(
            tmp_path,
            _manifest_dict(ed1_test=("HumanEval/3",)),
        )
    )
    exp_a = build_ed1_experiment(tasks=pool, split_manifest=a.for_env("ed1"))
    exp_b = build_ed1_experiment(tasks=pool, split_manifest=b.for_env("ed1"))
    ha = exp_a.eval_configs.internal.eval_config.config_identity_hash
    hb = exp_b.eval_configs.internal.eval_config.config_identity_hash
    # Different manifest content -> different tag -> different internal hash
    # even though internal ids happen to match.
    assert ha != hb


# --- CLI wiring: refusals + resolution ---------------------------------------


def test_cli_manifest_and_task_filter_mutually_exclusive(
    tmp_path: Path,
) -> None:
    p = _write(tmp_path, _manifest_dict())
    args = _cell_args(
        "ed1", task_split_manifest=str(p), task_filter="somefile.json"
    )
    with pytest.raises(ValueError, match="mutually"):
        _build_cell_config(args)


def test_cli_manifest_refused_for_ed1m(tmp_path: Path) -> None:
    p = _write(tmp_path, _manifest_dict())
    args = _cell_args("ed1m", task_split_manifest=str(p))
    with pytest.raises(TaskSplitManifestError, match="ed1m"):
        _build_cell_config(args)


def test_cli_manifest_resolves_roles_onto_config(tmp_path: Path) -> None:
    p = _write(tmp_path, _manifest_dict())
    args = _cell_args("ed1", task_split_manifest=str(p))
    config, _ = _build_cell_config(args)
    assert config.task_split_roles is not None
    assert config.task_split_roles.pool_key == "ed1"
    assert config.ed1_exclude_task_ids is None
