"""The dataset-specific EnvTask type playing the dr-code Task role.

Following dr-code's own convention (``humaneval_task_identity``: the Task
*role* is played by dataset-specific types with **no generic Task
superclass**), each whetstone-env instance is wrapped by an
:class:`EnvTask`: a frozen value carrying a *stable task identity* plus the
two field families the Rollout Definition consumes.

* **Graph External Inputs** -- the rendered prompt inputs a probe template
  consumes (``EnvTask.prompt_inputs``, keyed ``task.<field>``). These are the
  public inputs the LLM Call Node's prompt template renders against; they
  never include gold/oracle-only state.
* **Evaluation inputs** -- the gold/oracle-checkable state the terminal Eval
  Node's oracle operator scores against (``EnvTask.gold``). This is the
  instance's public ``gold`` field: the expected answer string for the
  re-derive envs, or the serialized constraint stack the c22 oracle re-runs.

The task identity is a full dr-serialize Identity Hash over the instance's
identity-defining fields: its ``(id, seed)`` pins, its content hash (a
SHA-256 over the instance's canonical fields, via
``whetstone_envs.core.content_hash`` applied to a one-instance pool), its
strata, and the env name. Two instances that are field-for-field identical
hash equal; any change to a prompt input, gold, seed, or id changes the
identity. Identity documents are produced through dr-serialize, the one
canonical identity lane.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr
from whetstone_envs.core import Instance, TaskPool, content_hash

from whetstone.optimization.identity import compute_identity_hash

ENV_TASK_SCHEMA = "whetstone.env_task"
ENV_TASK_SCHEMA_VERSION = 1

#: The Graph External Input prefix. A probe field ``constraints_block`` is
#: bound as the external input ``task.constraints_block`` on the LLM Call
#: Node, so prompt inputs and evaluation inputs share one ``task.`` namespace.
EXTERNAL_INPUT_PREFIX = "task."


def _instance_content_hash(instance: Instance) -> str:
    """The env content hash of a single instance (a one-instance pool).

    Reuses ``whetstone_envs.core.content_hash`` -- the same order-independent
    canonical-JSON SHA-256 the env manifests pin -- so an EnvTask's identity
    tracks exactly the fields the env repo treats as content-defining.
    """
    return content_hash(TaskPool((instance,)))


class EnvTask(BaseModel):
    """One whetstone-env instance wrapped as a dr-code Task-role value.

    Frozen. Carries the stable task identity, the env name, the Graph
    External Inputs (rendered prompt inputs), and the evaluation input
    (gold). There is deliberately no generic Task superclass: this is a
    dataset-specific Task-role type, exactly as dr-code's ``HumanEvalTask``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_name: StrictStr
    instance_id: StrictStr
    seed: StrictInt
    strata: tuple[StrictStr, ...]
    #: Graph External Inputs: the rendered prompt inputs, ``task.<field>``.
    prompt_inputs: tuple[tuple[StrictStr, StrictStr], ...]
    #: Evaluation input: the gold/oracle-checkable state.
    gold: StrictStr
    #: The env content hash of the wrapped instance (identity-bearing).
    instance_content_hash: StrictStr

    @classmethod
    def from_instance(cls, env_name: str, instance: Instance) -> EnvTask:
        """Wrap a whetstone-env :class:`Instance` for env ``env_name``."""
        return cls(
            env_name=env_name,
            instance_id=instance.id,
            seed=instance.seed,
            strata=tuple(instance.strata),
            prompt_inputs=tuple(
                (str(k), str(v))
                for k, v in sorted(instance.prompt_inputs.items())
            ),
            gold=instance.gold,
            instance_content_hash=_instance_content_hash(instance),
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
            "instance_id": self.instance_id,
            "seed": self.seed,
            "strata": list(self.strata),
            "instance_content_hash": self.instance_content_hash,
        }

    def task_identity(self) -> str:
        """The stable full Identity Hash of this task (dr-serialize)."""
        return compute_identity_hash(
            schema=ENV_TASK_SCHEMA,
            schema_version=ENV_TASK_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


__all__ = [
    "ENV_TASK_SCHEMA",
    "ENV_TASK_SCHEMA_VERSION",
    "EXTERNAL_INPUT_PREFIX",
    "EnvTask",
]
