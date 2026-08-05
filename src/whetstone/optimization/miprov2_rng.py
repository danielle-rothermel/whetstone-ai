"""Serializable shared-RNG state for the durable MIPROv2 control flow."""

from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from whetstone.optimization.identity import (
    ImmutableJsonObject,
    require_full_hash,
)

Miprov2RngOperation = Literal["sample", "shuffle", "choice", "randint"]
MIPROV2_DEMO_BRIDGE_VERSION = "whetstone_component_demo_bridge/v1"


class Miprov2RandomState(BaseModel):
    """JSON-safe snapshot of Python's MT19937 state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: StrictInt
    internal: tuple[StrictInt, ...]
    gaussian_next: float | None = None

    @classmethod
    def seeded(cls, seed: int) -> Miprov2RandomState:
        return cls.from_random(random.Random(seed))

    @classmethod
    def from_random(cls, rng: random.Random) -> Miprov2RandomState:
        version, internal, gaussian_next = rng.getstate()
        return cls(
            version=version,
            internal=tuple(internal),
            gaussian_next=gaussian_next,
        )

    def restore(self) -> random.Random:
        rng = random.Random()
        try:
            rng.setstate((self.version, self.internal, self.gaussian_next))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid serialized MIPROv2 RNG state") from exc
        return rng


class Miprov2RngDraw(BaseModel):
    """One ordered, replay-verifiable draw from MIPROv2's shared RNG."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: StrictInt
    phase: Literal["dataset", "bootstrap", "proposal", "evaluation"]
    operation: Miprov2RngOperation
    arguments: tuple[Any, ...] = ()
    result: Any

    @field_validator("arguments", "result", mode="before")
    @classmethod
    def _normalize_json_sequences(cls, value: Any) -> Any:
        return _lists_for_tuples(value)

    @model_validator(mode="after")
    def _validate_draw(self) -> Miprov2RngDraw:
        if self.ordinal < 0:
            raise ValueError("RNG draw ordinal cannot be negative")
        dumped = self.model_dump(mode="json")
        ImmutableJsonObject({"value": dumped["arguments"]})
        ImmutableJsonObject({"value": dumped["result"]})
        return self


class Miprov2RngCheckpoint(BaseModel):
    """The one durable shared-RNG cursor and its append-only transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: StrictInt
    state: Miprov2RandomState
    draws: tuple[Miprov2RngDraw, ...] = ()

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> Miprov2RngCheckpoint:
        if tuple(draw.ordinal for draw in self.draws) != tuple(
            range(len(self.draws))
        ):
            raise ValueError("RNG draw ordinals must be contiguous from zero")
        # Replay from the originating seed.  A serialized terminal MT state and
        # a self-consistent-looking draw list are not independently trusted.
        rng = random.Random(self.seed)
        for draw in self.draws:
            actual = _replay_draw(rng, draw)
            if _lists_for_tuples(actual) != draw.result:
                raise ValueError(
                    f"RNG draw {draw.ordinal} result does not match replay"
                )
        if Miprov2RandomState.from_random(rng) != self.state:
            raise ValueError(
                "serialized MIPROv2 RNG state does not match seed and draws"
            )
        return self

    @classmethod
    def seeded(cls, seed: int) -> Miprov2RngCheckpoint:
        return cls(seed=seed, state=Miprov2RandomState.seeded(seed))

    @classmethod
    def after_validation_sampling(
        cls,
        *,
        seed: int,
        population_size: int,
        sample_indices: tuple[int, ...] | None,
    ) -> Miprov2RngCheckpoint:
        """Reconstruct the shared cursor from resolved MIPROv2 control.

        Auto mode records its ordered validation sample in control; manual
        mode supplies ``None`` and consumes no draw.
        """

        rng = random.Random(seed)
        checkpoint = cls.seeded(seed)
        if sample_indices is None:
            return checkpoint
        sampled = tuple(
            rng.sample(range(population_size), len(sample_indices))
        )
        if sampled != sample_indices:
            raise ValueError(
                "validation sample does not match the shared MIPROv2 RNG"
            )
        return checkpoint.append(
            rng=rng,
            phase="dataset",
            operation="sample",
            arguments=(population_size, len(sample_indices)),
            result=sampled,
        )

    @classmethod
    def from_random(
        cls,
        rng: random.Random,
        *,
        seed: int,
        draws: tuple[Miprov2RngDraw, ...] = (),
    ) -> Miprov2RngCheckpoint:
        return cls(
            seed=seed,
            state=Miprov2RandomState.from_random(rng),
            draws=draws,
        )

    def append(
        self,
        *,
        rng: random.Random,
        phase: Literal["dataset", "bootstrap", "proposal", "evaluation"],
        operation: Miprov2RngOperation,
        arguments: tuple[Any, ...],
        result: Any,
    ) -> Miprov2RngCheckpoint:
        draw = Miprov2RngDraw(
            ordinal=len(self.draws),
            phase=phase,
            operation=operation,
            arguments=arguments,
            result=result,
        )
        return Miprov2RngCheckpoint(
            seed=self.seed,
            state=Miprov2RandomState.from_random(rng),
            draws=(*self.draws, draw),
        )


class Miprov2DurableBindings(BaseModel):
    """Immutable authorities shared by every MIPROv2 pure request and state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_identity_hash: StrictStr
    prompt_route_identity_hash: StrictStr
    task_route_identity_hash: StrictStr
    execution_policy_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    base_candidate_identity_hash: StrictStr
    teacher_candidate_identity_hash: StrictStr
    demo_bridge_version: Literal["whetstone_component_demo_bridge/v1"] = (
        MIPROV2_DEMO_BRIDGE_VERSION
    )

    @model_validator(mode="after")
    def _validate_hashes(self) -> Miprov2DurableBindings:
        for field in (
            "control_identity_hash",
            "prompt_route_identity_hash",
            "task_route_identity_hash",
            "execution_policy_identity_hash",
            "prompt_adapter_identity_hash",
            "base_candidate_identity_hash",
            "teacher_candidate_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        return self


def _replay_draw(rng: random.Random, draw: Miprov2RngDraw) -> Any:
    """Execute one draw with its exact Python ``Random`` semantics."""

    allowed = {
        "dataset": {"sample"},
        "bootstrap": {"shuffle", "randint"},
        "proposal": {"choice", "randint"},
        "evaluation": {"sample"},
    }
    if draw.operation not in allowed[draw.phase]:
        raise ValueError(
            f"RNG operation {draw.operation!r} is invalid in {draw.phase!r}"
        )
    arguments = draw.arguments
    if draw.operation == "sample":
        if len(arguments) != 2:
            raise ValueError("sample draw requires population and count")
        population_spec, count = arguments
        if type(count) is not int:
            raise ValueError("sample count must be an integer")
        if type(population_spec) is int:
            if population_spec < 0:
                raise ValueError("sample population size cannot be negative")
            population: range | list[Any] = range(population_spec)
        elif isinstance(population_spec, (list, tuple)):
            population = list(population_spec)
        else:
            raise ValueError("sample population must be a size or sequence")
        return tuple(rng.sample(population, count))
    if draw.operation == "shuffle":
        if len(arguments) != 1 or not isinstance(arguments[0], (list, tuple)):
            raise ValueError("shuffle draw requires one ordered sequence")
        shuffled = list(arguments[0])
        rng.shuffle(shuffled)
        return tuple(shuffled)
    if draw.operation == "choice":
        if not arguments:
            raise ValueError("choice draw requires a nonempty sequence")
        return rng.choice(arguments)
    if len(arguments) != 2 or any(
        type(value) is not int for value in arguments
    ):
        raise ValueError("randint draw requires two integer bounds")
    return rng.randint(arguments[0], arguments[1])


def _lists_for_tuples(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_lists_for_tuples(item) for item in value]
    if isinstance(value, list):
        return [_lists_for_tuples(item) for item in value]
    if isinstance(value, dict):
        return {key: _lists_for_tuples(item) for key, item in value.items()}
    return value


__all__ = [
    "MIPROV2_DEMO_BRIDGE_VERSION",
    "Miprov2DurableBindings",
    "Miprov2RandomState",
    "Miprov2RngCheckpoint",
    "Miprov2RngDraw",
    "Miprov2RngOperation",
]
