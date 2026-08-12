from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from whetstone.core.identity import (
    ImmutableJsonObject,
    require_full_hash,
)

Miprov2RngOperation = Literal["sample", "shuffle", "choice", "randint"]
MIPROV2_DEMO_BRIDGE_VERSION = "whetstone_component_demo_bridge/v1"


class Miprov2RandomState(BaseModel):
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: StrictInt
    phase: Literal["dataset", "bootstrap", "proposal", "evaluation"]
    operation: Miprov2RngOperation
    arguments: tuple[Any, ...] = ()
    result: Any

    @field_validator("arguments", "result", mode="before")
    @classmethod
    def _freeze_json(cls, value: Any) -> Any:
        return _freeze_json_value(value)

    @field_serializer("arguments", "result")
    def _serialize_json(self, value: Any) -> Any:
        return _json_value(value)

    def model_post_init(self, _context: Any) -> None:
        object.__setattr__(
            self,
            "arguments",
            tuple(_freeze_json_value(item) for item in self.arguments),
        )
        object.__setattr__(self, "result", _freeze_json_value(self.result))

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if deep:
            payload = self.model_dump(mode="json")
            payload.update(update or {})
            return type(self).model_validate(payload)
        copied = super().model_copy(update=update, deep=deep)
        copied.model_post_init(None)
        return copied

    @model_validator(mode="after")
    def _validate_draw(self) -> Miprov2RngDraw:
        if self.ordinal < 0:
            raise ValueError("RNG draw ordinal cannot be negative")
        return self


class Miprov2RngCheckpoint(BaseModel):
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

        rng = random.Random(self.seed)
        for draw in self.draws:
            actual = _replay_draw(rng, draw)
            if _lists_for_tuples(actual) != _lists_for_tuples(draw.result):
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    control_identity_hash: StrictStr
    prompt_route_identity_hash: StrictStr
    task_route_identity_hash: StrictStr
    execution_policy_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    proposal_executor_policy_identity_hash: StrictStr
    proposal_transport_durability_identity_hash: StrictStr
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
            "proposal_executor_policy_identity_hash",
            "proposal_transport_durability_identity_hash",
            "base_candidate_identity_hash",
            "teacher_candidate_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        return self


def _replay_draw(rng: random.Random, draw: Miprov2RngDraw) -> Any:

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
    if isinstance(value, ImmutableJsonObject):
        return value.to_json()
    if isinstance(value, tuple):
        return [_lists_for_tuples(item) for item in value]
    if isinstance(value, list):
        return [_lists_for_tuples(item) for item in value]
    if isinstance(value, dict):
        return {key: _lists_for_tuples(item) for key, item in value.items()}
    return value


def _freeze_json_value(value: Any) -> Any:

    if isinstance(value, ImmutableJsonObject):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    wrapper = ImmutableJsonObject({"value": value})
    return wrapper["value"]


def _json_value(value: Any) -> Any:
    if isinstance(value, ImmutableJsonObject):
        return value.to_json()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "MIPROV2_DEMO_BRIDGE_VERSION",
    "Miprov2DurableBindings",
    "Miprov2RandomState",
    "Miprov2RngCheckpoint",
    "Miprov2RngDraw",
    "Miprov2RngOperation",
]
