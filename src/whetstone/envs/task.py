from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from whetstone_envs.core import Instance, TaskPool, content_hash

from whetstone.core.identity import compute_identity_hash

TASK_SCHEMA = "whetstone.task"
TASK_SCHEMA_VERSION = 1

#: The Graph External Input prefix. A probe field ``constraints_block`` is
#: bound as the external input ``task.constraints_block`` on the LLM Call
#: Node, so prompt inputs and evaluation inputs share one ``task.`` namespace.
EXTERNAL_INPUT_PREFIX = "task."


def _task_content_hash(instance: Instance) -> str:
    """The env content hash of a single instance (a one-instance pool).

    Reuses ``whetstone_envs.core.content_hash`` -- the same order-independent
    canonical-JSON SHA-256 the env manifests pin -- so an Task's identity
    tracks exactly the fields the env repo treats as content-defining.
    """
    return content_hash(TaskPool((instance,)))


@dataclass(frozen=True, slots=True)
class Task:
    """One whetstone-env instance wrapped as a Task-role value.

    Frozen. Carries the stable task identity, the env name, the Graph
    External Inputs (rendered prompt inputs), and the evaluation input
    (gold). There is deliberately no generic Task superclass: this is a
    dataset-specific Task-role type, like ``HumanEvalTask`` for code tasks.
    """

    env_name: str
    task_id: str
    seed: int
    strata: tuple[str, ...]
    #: Graph External Inputs: the rendered prompt inputs, ``task.<field>``.
    prompt_inputs: tuple[tuple[str, str], ...]
    #: Evaluation input: the gold/oracle-checkable state.
    gold: str
    #: The env content hash of the wrapped instance (identity-bearing).
    task_content_hash: str

    @classmethod
    def from_instance(cls, env_name: str, instance: Instance) -> Task:
        """Wrap a whetstone-env :class:`Instance` for env ``env_name``."""
        return cls(
            env_name=env_name,
            task_id=instance.id,
            seed=instance.seed,
            strata=tuple(instance.strata),
            prompt_inputs=tuple(
                (str(k), str(v))
                for k, v in sorted(instance.prompt_inputs.items())
            ),
            gold=instance.gold,
            task_content_hash=_task_content_hash(instance),
        )

    @property
    def stratum(self) -> str:
        """The instance's primary (first) stratum label."""
        return self.strata[0]

    def prompt_inputs_dict(self) -> dict[str, str]:
        """The Graph External Inputs as an ordered mapping."""
        return {k: v for k, v in self.prompt_inputs}

    def external_input_fields(self) -> tuple[str, ...]:
        """The ``task.<field>`` external-input names this task supplies."""
        return tuple(
            f"{EXTERNAL_INPUT_PREFIX}{k}" for k, _ in self.prompt_inputs
        )

    def identity_payload(self) -> dict[str, Any]:
        """The identity-defining payload (ordering-stable, JSON-safe)."""
        return {
            "env_name": self.env_name,
            "task_id": self.task_id,
            "seed": self.seed,
            "strata": list(self.strata),
            "task_content_hash": self.task_content_hash,
        }

    def task_hash(self) -> str:
        """The stable full Identity Hash of this task (dr-serialize)."""
        return compute_identity_hash(
            schema=TASK_SCHEMA,
            schema_version=TASK_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


__all__ = [
    "EXTERNAL_INPUT_PREFIX",
    "TASK_SCHEMA",
    "TASK_SCHEMA_VERSION",
    "Task",
]
