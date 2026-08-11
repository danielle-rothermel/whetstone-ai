from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from dr_providers import ProviderCallConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from whetstone.core.identity import IdentityHash, compute_identity_hash
from whetstone.envs.code_comp.constants import (
    CODE_COMP_CANONICAL_MODEL,
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_DEFAULT_BUDGET_RATIO,
    CODE_COMP_ENV_NAME,
)
from whetstone.envs.code_comp.dataset import CodeCompTaskInstance
from whetstone.envs.code_comp.generation_graph.direct import (
    DIRECT_DEFAULT_RENAME_TOKEN,
    DIRECT_INPUT_ARMS,
)
from whetstone.envs.code_comp.generation_graph.encdec import (
    build_encoder_provider_call_config,
)
from whetstone.envs.code_comp.mode import CodeCompMode
from whetstone.envs.code_comp.procedure import (
    build_encdec_procedure_config,
)
from whetstone.envs.code_comp.reward.blended import (
    CODE_COMP_DEFAULT_BLEND_CONFIG,
    BoundedCompressionMetricConfig,
)
from whetstone.envs.code_comp.submission_result import CodeSubmissionResult
from whetstone.envs.sampling import Completeness
from whetstone.evaluation import EvaluationProcedureConfig
from whetstone.experiment.task_selection import TaskSplitRoles

if TYPE_CHECKING:
    from dr_code.humaneval import HumanEvalTask
    from whetstone_envs.core import Instance

    from whetstone.envs.code_comp.experiment import CodeCompExperiment

CODE_COMP_EXPERIMENT_CONFIG_SCHEMA = "whetstone.code_comp.experiment_config"
CODE_COMP_EXPERIMENT_CONFIG_SCHEMA_VERSION = 1


class CompressionOperatorConfig(BaseModel):
    """Identity-bearing compression operator settings for procedure metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["zstd"] = "zstd"
    level: int = Field(default=19, ge=1)


class CodeCompModelRouteConfig(BaseModel):
    """One encoder or decoder provider route."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = None
    provider_call_config: ProviderCallConfig | None = None

    @model_validator(mode="after")
    def _exactly_one_route(self) -> Self:
        if (self.model is None) == (self.provider_call_config is None):
            raise ValueError(
                "exactly one of model or provider_call_config is required"
            )
        return self

    def resolve(self) -> ProviderCallConfig:
        if self.provider_call_config is not None:
            return self.provider_call_config
        assert self.model is not None
        return build_encoder_provider_call_config(self.model)

    def identity_payload(self) -> dict[str, str]:
        if self.provider_call_config is not None:
            return {
                "kind": "provider_call_config",
                "identity_hash": self.provider_call_config.identity_hash,
            }
        assert self.model is not None
        return {"kind": "model", "model": self.model}


class CodeCompModelRoutesConfig(BaseModel):
    """Encoder and optional distinct decoder routes (``M_e`` / ``M_d``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoder: CodeCompModelRouteConfig
    decoder: CodeCompModelRouteConfig | None = None

    def decoder_route(self) -> CodeCompModelRouteConfig:
        return self.decoder if self.decoder is not None else self.encoder

    def encoder_call_config(self) -> ProviderCallConfig:
        return self.encoder.resolve()

    def decoder_call_config(self) -> ProviderCallConfig:
        return self.decoder_route().resolve()


class CodeCompPoolConfig(BaseModel):
    """Task pool selection for HumanEval-backed modes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_path: Path | None = None
    limit: int | None = Field(default=None, ge=1)
    tasks: tuple[CodeCompTaskInstance, ...] | None = None


class CodeCompSplitConfig(BaseModel):
    """Internal / official split semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    internal_n: int | None = Field(default=None, ge=1)
    official_n: int | None = Field(default=None, ge=1)
    split_manifest: TaskSplitRoles | None = None
    exclude_task_ids: frozenset[str] = frozenset()


class CodeCompSamplingConfig(BaseModel):
    """Sampling and completeness policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_samples: int = Field(default=3, ge=1)
    completeness: Completeness = Completeness.PROPAGATE
    max_skip_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class DirectModeSettings(BaseModel):
    """Mode-specific settings for ``direct`` experiments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_arm: str = "original"
    rename_token: str = DIRECT_DEFAULT_RENAME_TOKEN
    model: str = CODE_COMP_CANONICAL_MODEL

    @model_validator(mode="after")
    def _validate_arm(self) -> Self:
        if self.input_arm not in DIRECT_INPUT_ARMS:
            raise ValueError(
                f"unknown direct input arm {self.input_arm!r} "
                f"(choose one of {DIRECT_INPUT_ARMS})"
            )
        return self


class EncDecModeSettings(BaseModel):
    """Mode-specific settings for ``encdec`` experiments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_ratio: float | None = CODE_COMP_DEFAULT_BUDGET_RATIO
    blend_config: BoundedCompressionMetricConfig = Field(
        default_factory=BoundedCompressionMetricConfig
    )


class MutantModeSettings(BaseModel):
    """Mode-specific settings for ``encdec_mutant`` experiments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_dir: Path
    exclude_mutant_ids: frozenset[str] = frozenset()
    budget_ratio: float | None = None
    blend_config: BoundedCompressionMetricConfig | None = None


class CodeCompExperimentConfig(BaseModel):
    """Identity-bearing input that materializes one code_comp experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: CodeCompMode
    models: CodeCompModelRoutesConfig
    compression: CompressionOperatorConfig = Field(
        default_factory=CompressionOperatorConfig
    )
    pool: CodeCompPoolConfig = Field(default_factory=CodeCompPoolConfig)
    split: CodeCompSplitConfig = Field(default_factory=CodeCompSplitConfig)
    sampling: CodeCompSamplingConfig = Field(
        default_factory=CodeCompSamplingConfig
    )
    direct: DirectModeSettings | None = None
    encdec: EncDecModeSettings | None = None
    mutant: MutantModeSettings | None = None

    @model_validator(mode="after")
    def _mode_settings(self) -> Self:
        settings = {
            CodeCompMode.DIRECT: self.direct,
            CodeCompMode.ENCDEC: self.encdec,
            CodeCompMode.ENCDEC_MUTANT: self.mutant,
        }
        active = settings[self.mode]
        if active is None:
            raise ValueError(f"mode {self.mode.value!r} requires its settings")
        for other_mode, other_settings in settings.items():
            if other_mode is not self.mode and other_settings is not None:
                raise ValueError(
                    f"mode {self.mode.value!r} cannot carry "
                    f"{other_mode.value!r} settings"
                )
        if self.mode is CodeCompMode.DIRECT and self.pool.tasks is None:
            if self.pool.snapshot_path is None and self.pool.limit is None:
                pass  # load full pinned pool at build time
        if self.mode is CodeCompMode.ENCDEC_MUTANT:
            assert self.mutant is not None
        return self

    def identity_hash(self) -> IdentityHash:
        return compute_identity_hash(
            schema=CODE_COMP_EXPERIMENT_CONFIG_SCHEMA,
            schema_version=CODE_COMP_EXPERIMENT_CONFIG_SCHEMA_VERSION,
            payload=self._identity_payload(),
        )

    def _identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode.value,
            "models": {
                "encoder": self.models.encoder.identity_payload(),
                "decoder": (
                    self.models.decoder.identity_payload()
                    if self.models.decoder is not None
                    else None
                ),
            },
            "compression": self.compression.model_dump(mode="json"),
            "pool": {
                "snapshot_path": (
                    str(self.pool.snapshot_path)
                    if self.pool.snapshot_path is not None
                    else None
                ),
                "limit": self.pool.limit,
                "task_ids": (
                    [str(t.instance.id) for t in self.pool.tasks]
                    if self.pool.tasks is not None
                    else None
                ),
            },
            "split": {
                "internal_n": self.split.internal_n,
                "official_n": self.split.official_n,
                "split_manifest": (
                    {
                        "pool_key": self.split.split_manifest.pool_key,
                        "train_ids": list(self.split.split_manifest.train_ids),
                        "val_ids": list(self.split.split_manifest.val_ids),
                        "test_ids": list(self.split.split_manifest.test_ids),
                        "content_hash": self.split.split_manifest.content_hash,
                    }
                    if self.split.split_manifest is not None
                    else None
                ),
                "exclude_task_ids": sorted(self.split.exclude_task_ids),
            },
            "sampling": self.sampling.model_dump(mode="json"),
        }
        if self.direct is not None:
            payload["direct"] = self.direct.model_dump(mode="json")
        if self.encdec is not None:
            payload["encdec"] = {
                "budget_ratio": self.encdec.budget_ratio,
                "blend_config": self.encdec.blend_config.model_dump(
                    mode="json"
                ),
            }
        if self.mutant is not None:
            payload["mutant"] = {
                "artifact_dir": str(self.mutant.artifact_dir),
                "exclude_mutant_ids": sorted(self.mutant.exclude_mutant_ids),
                "budget_ratio": self.mutant.budget_ratio,
                "blend_config": (
                    self.mutant.blend_config.model_dump(mode="json")
                    if self.mutant.blend_config is not None
                    else None
                ),
            }
        return payload

    def build_procedure_config(self) -> EvaluationProcedureConfig:
        """Materialize the evaluation procedure for this config's mode."""
        if self.mode is CodeCompMode.ENCDEC_MUTANT:
            from whetstone.envs.code_comp.modes.mutant import (
                build_mutant_procedure_config,
            )

            return build_mutant_procedure_config()
        return build_encdec_procedure_config(
            compression_level=self.compression.level,
        )

    def build_generation_graph(
        self,
        *,
        procedure_config_hash: str,
    ):
        """Materialize the encdec generation graph for encdec-family modes."""
        from whetstone.envs.code_comp.generation_graph.encdec import (
            build_encdec_generation_graph,
        )

        if self.mode is CodeCompMode.DIRECT:
            from whetstone.envs.code_comp.generation_graph.direct import (
                build_direct_generation_graph,
            )

            assert self.direct is not None
            return build_direct_generation_graph(
                model=self.direct.model,
                procedure_config_hash=procedure_config_hash,
                input_arm=self.direct.input_arm,
                rename_token=self.direct.rename_token,
            )
        budget_ratio: float | None
        if self.mode is CodeCompMode.ENCDEC:
            assert self.encdec is not None
            budget_ratio = self.encdec.budget_ratio
        else:
            assert self.mutant is not None
            budget_ratio = self.mutant.budget_ratio
        return build_encdec_generation_graph(
            CODE_COMP_ENV_NAME,
            encoder_call_config=self.models.encoder_call_config(),
            decoder_call_config=self.models.decoder_call_config(),
            procedure_config_hash=procedure_config_hash,
            budget_ratio=budget_ratio,
        )

    def build_experiment(
        self,
        *,
        scorer: Callable[..., CodeSubmissionResult] | None = None,
    ) -> CodeCompExperiment:
        """Materialize one experiment from this config."""
        from whetstone.envs.code_comp.experiment import (
            DirectExperiment,
            EncDecExperiment,
            MutantExperiment,
            validate_encdec_blend,
        )

        procedure = self.build_procedure_config()
        generation_graph = self.build_generation_graph(
            procedure_config_hash=procedure.config_hash,
        )
        policy = self.sampling.completeness.to_policy(
            max_skip_fraction=self.sampling.max_skip_fraction
        )
        if self.mode is CodeCompMode.DIRECT:
            from whetstone.envs.code_comp.modes.direct import (
                _direct_split,
                build_direct_reward_policy,
                direct_ceiling_candidate,
                direct_initial_candidate,
            )

            assert self.direct is not None
            pool, humaneval_by_id = self._load_direct_pool()
            internal_instances, official_tasks, manifest_tag = (
                self._resolve_split(pool)
            )
            internal_split = _direct_split(
                split_role="internal_eval",
                tasks=internal_instances,
                procedure=procedure,
                completeness=self.sampling.completeness,
                max_skip_fraction=self.sampling.max_skip_fraction,
                num_samples=self.sampling.num_samples,
                input_arm=self.direct.input_arm,
                rename_token=self.direct.rename_token,
                manifest_tag=manifest_tag,
            )
            official_split = _direct_split(
                split_role="official",
                tasks=official_tasks,
                procedure=procedure,
                completeness=self.sampling.completeness,
                max_skip_fraction=self.sampling.max_skip_fraction,
                num_samples=self.sampling.num_samples,
                input_arm=self.direct.input_arm,
                rename_token=self.direct.rename_token,
                manifest_tag=manifest_tag,
            )
            from whetstone.envs.sampling import EnvEvalConfigs

            eval_configs = EnvEvalConfigs(
                env_name=CODE_COMP_ENV_NAME,
                procedure_config_hash=procedure.config_hash,
                internal=internal_split,
                official=official_split,
                held_out_task_hashes=(),
            )
            return DirectExperiment(
                config=self,
                env_name=CODE_COMP_ENV_NAME,
                generation_graph=generation_graph,
                initial_candidate=direct_initial_candidate(),
                ceiling_candidate=direct_ceiling_candidate(),
                eval_configs=eval_configs,
                reward_policy=build_direct_reward_policy(),
                completeness_policy=policy,
                scorer=scorer,
                humaneval_by_id=humaneval_by_id,
            )
        if self.mode is CodeCompMode.ENCDEC:
            from whetstone.envs.code_comp.modes.encdec import (
                _code_comp_split,
                encdec_ceiling_candidate,
                encdec_initial_candidate,
            )
            from whetstone.envs.code_comp.reward.blended import (
                build_code_comp_blended_reward_policy,
            )
            from whetstone.envs.sampling import EnvEvalConfigs

            assert self.encdec is not None
            pool = self._load_task_pool()
            internal_instances, official_tasks, manifest_tag = (
                self._resolve_split(pool)
            )
            internal_split = _code_comp_split(
                dataset_revision=CODE_COMP_DATASET_REVISION,
                split_role="internal_eval",
                tasks=internal_instances,
                procedure=procedure,
                completeness=self.sampling.completeness,
                max_skip_fraction=self.sampling.max_skip_fraction,
                num_samples=self.sampling.num_samples,
                manifest_tag=manifest_tag,
            )
            official_split = _code_comp_split(
                dataset_revision=CODE_COMP_DATASET_REVISION,
                split_role="official",
                tasks=official_tasks,
                procedure=procedure,
                completeness=self.sampling.completeness,
                max_skip_fraction=self.sampling.max_skip_fraction,
                num_samples=self.sampling.num_samples,
                manifest_tag=manifest_tag,
            )
            eval_configs = EnvEvalConfigs(
                env_name=CODE_COMP_ENV_NAME,
                procedure_config_hash=procedure.config_hash,
                internal=internal_split,
                official=official_split,
                held_out_task_hashes=(),
            )
            experiment = EncDecExperiment(
                config=self,
                env_name=CODE_COMP_ENV_NAME,
                generation_graph=generation_graph,
                initial_candidate=encdec_initial_candidate(),
                ceiling_candidate=encdec_ceiling_candidate(),
                eval_configs=eval_configs,
                reward_policy=build_code_comp_blended_reward_policy(
                    self.encdec.blend_config,
                    env_name=CODE_COMP_ENV_NAME,
                ),
                completeness_policy=policy,
                encdec_generation_graph=generation_graph,
                dataset_revision=CODE_COMP_DATASET_REVISION,
                scorer=scorer,
                blend_config=self.encdec.blend_config,
            )
            validate_encdec_blend(experiment)
            return experiment
        from whetstone.envs.code_comp.modes.encdec import (
            _code_comp_split,
            encdec_ceiling_candidate,
            encdec_initial_candidate,
        )
        from whetstone.envs.code_comp.modes.mutant import (
            _mutant_to_instance,
            build_mutant_reward_policy,
        )
        from whetstone.envs.code_comp.mutant.dataset import load_dataset
        from whetstone.envs.code_comp.reward.blended import (
            build_code_comp_blended_reward_policy,
        )
        from whetstone.envs.sampling import EnvEvalConfigs

        assert self.mutant is not None
        loaded = load_dataset(self.mutant.artifact_dir)
        records = loaded.records
        if self.pool.limit is not None:
            records = records[: self.pool.limit]
        if self.mutant.exclude_mutant_ids:
            records = tuple(
                mutant
                for mutant in records
                if mutant.content_hash not in self.mutant.exclude_mutant_ids
            )
        if not records:
            raise ValueError("ed1m mutant pool is empty")
        all_instances = tuple(_mutant_to_instance(m) for m in records)
        mutant_map = {m.content_hash: m for m in records}
        n = len(all_instances)
        i_n = (
            self.split.internal_n
            if self.split.internal_n is not None
            else min(max(1, n // 2), n)
        )
        internal_instances = all_instances[:i_n]
        rest = all_instances[i_n:]
        o_n = (
            self.split.official_n
            if self.split.official_n is not None
            else len(rest)
        )
        official_tasks = rest[:o_n] if rest else internal_instances[: o_n or n]
        if not official_tasks:
            official_tasks = internal_instances
        internal_split = _code_comp_split(
            env_name=CODE_COMP_ENV_NAME,
            dataset_revision=loaded.manifest.dataset_hash,
            split_role="internal_eval",
            tasks=internal_instances,
            procedure=procedure,
            completeness=self.sampling.completeness,
            max_skip_fraction=self.sampling.max_skip_fraction,
            num_samples=self.sampling.num_samples,
        )
        official_split = _code_comp_split(
            env_name=CODE_COMP_ENV_NAME,
            dataset_revision=loaded.manifest.dataset_hash,
            split_role="official",
            tasks=official_tasks,
            procedure=procedure,
            completeness=self.sampling.completeness,
            max_skip_fraction=self.sampling.max_skip_fraction,
            num_samples=self.sampling.num_samples,
        )
        eval_configs = EnvEvalConfigs(
            env_name=CODE_COMP_ENV_NAME,
            procedure_config_hash=procedure.config_hash,
            internal=internal_split,
            official=official_split,
            held_out_task_hashes=(),
        )
        blend = self.mutant.blend_config
        return MutantExperiment(
            config=self,
            env_name=CODE_COMP_ENV_NAME,
            generation_graph=generation_graph,
            initial_candidate=encdec_initial_candidate(),
            ceiling_candidate=encdec_ceiling_candidate(),
            eval_configs=eval_configs,
            reward_policy=(
                build_mutant_reward_policy()
                if blend is None
                else build_code_comp_blended_reward_policy(
                    blend, env_name=CODE_COMP_ENV_NAME
                )
            ),
            completeness_policy=policy,
            encdec_generation_graph=generation_graph,
            dataset_revision=loaded.manifest.dataset_hash,
            scorer=scorer,
            blend_config=blend,
            mutants=mutant_map,
        )

    def _load_task_pool(self) -> tuple[CodeCompTaskInstance, ...]:
        from whetstone.envs.code_comp.dataset import load_tasks

        pool = (
            self.pool.tasks
            if self.pool.tasks is not None
            else load_tasks(
                snapshot_path=self.pool.snapshot_path,
                limit=self.pool.limit,
            )
        )
        if self.split.exclude_task_ids:
            pool = tuple(
                t
                for t in pool
                if str(t.instance.id) not in self.split.exclude_task_ids
            )
        if not pool:
            raise ValueError("code_comp task pool is empty")
        return pool

    def _load_direct_pool(
        self,
    ) -> tuple[tuple[CodeCompTaskInstance, ...], dict[str, HumanEvalTask]]:

        pool = self._load_task_pool()
        humaneval_by_id = {str(t.instance.id): t.humaneval_task for t in pool}
        return pool, humaneval_by_id

    def _resolve_split(
        self,
        pool: tuple[CodeCompTaskInstance, ...],
    ) -> tuple[tuple[Instance, ...], tuple[Instance, ...], str | None]:

        from whetstone.experiment.task_selection import resolve_manifest_split

        manifest_tag: str | None = None
        if self.split.split_manifest is not None:
            resolved = resolve_manifest_split(
                roles=self.split.split_manifest,
                items=pool,
                id_of=lambda t: str(t.instance.id),
                official_n=self.split.official_n,
            )
            internal_instances = tuple(t.instance for t in resolved.internal)
            official_tasks = tuple(t.instance for t in resolved.official)
            manifest_tag = resolved.manifest_tag
            if resolved.official_capped:
                print(f"[code_comp] {resolved.official_capped}")
            return internal_instances, official_tasks, manifest_tag
        all_instances = tuple(t.instance for t in pool)
        n = len(all_instances)
        i_n = (
            self.split.internal_n
            if self.split.internal_n is not None
            else min(max(1, n // 2), n)
        )
        internal_instances = all_instances[:i_n]
        rest = all_instances[i_n:]
        o_n = (
            self.split.official_n
            if self.split.official_n is not None
            else len(rest)
        )
        official_tasks = rest[:o_n] if rest else internal_instances[: o_n or n]
        if not official_tasks:
            official_tasks = internal_instances
        return internal_instances, official_tasks, manifest_tag

    def _builder_kwargs(
        self,
        *,
        scorer: Callable[..., CodeSubmissionResult] | None,
    ) -> dict[str, Any]:
        common = {
            "scorer": scorer,
            "internal_n": self.split.internal_n,
            "official_n": self.split.official_n,
            "completeness": self.sampling.completeness,
            "max_skip_fraction": self.sampling.max_skip_fraction,
            "num_samples": self.sampling.num_samples,
            "split_manifest": self.split.split_manifest,
        }
        if self.mode is CodeCompMode.DIRECT:
            assert self.direct is not None
            kwargs: dict[str, Any] = {
                **common,
                "model": self.direct.model,
                "input_arm": self.direct.input_arm,
                "rename_token": self.direct.rename_token,
            }
        elif self.mode is CodeCompMode.ENCDEC:
            assert self.encdec is not None
            kwargs = {
                **common,
                "provider_call_config": self.models.encoder_call_config(),
                "budget_ratio": self.encdec.budget_ratio,
                "blend_config": self.encdec.blend_config,
            }
        else:
            assert self.mutant is not None
            kwargs = {
                "internal_n": self.split.internal_n,
                "official_n": self.split.official_n,
                "completeness": self.sampling.completeness,
                "max_skip_fraction": self.sampling.max_skip_fraction,
                "num_samples": self.sampling.num_samples,
                "scorer": scorer,
                "artifact_dir": self.mutant.artifact_dir,
                "provider_call_config": self.models.encoder_call_config(),
                "budget_ratio": self.mutant.budget_ratio,
                "exclude_mutant_ids": self.mutant.exclude_mutant_ids or None,
                "blend_config": self.mutant.blend_config,
            }
            if self.pool.limit is not None:
                kwargs["limit"] = self.pool.limit
            return kwargs
        if self.pool.tasks is not None:
            kwargs["tasks"] = self.pool.tasks
        else:
            kwargs["snapshot_path"] = self.pool.snapshot_path
            kwargs["limit"] = self.pool.limit
        if self.split.exclude_task_ids:
            kwargs["exclude_task_ids"] = self.split.exclude_task_ids
        return kwargs

    def preview_metadata(self) -> dict[str, str]:
        """Return stable preview metadata keyed by config identity."""
        return {
            "mode": self.mode.value,
            "config_hash": str(self.identity_hash()),
            "dataset_revision": CODE_COMP_DATASET_REVISION,
            "env_name": CODE_COMP_ENV_NAME,
        }


def default_code_comp_config(
    mode: CodeCompMode,
    /,
    **overrides: Any,
) -> CodeCompExperimentConfig:
    """Build the canonical default config for one mode.

    Optional ``overrides`` may replace nested sub-config fields.
    """
    artifact_dir = None
    if mode is CodeCompMode.ENCDEC_MUTANT:
        artifact_dir = overrides.pop("artifact_dir", None)
        if artifact_dir is None:
            raise ValueError("encdec_mutant requires artifact_dir")
    encoder = CodeCompModelRouteConfig(model=CODE_COMP_CANONICAL_MODEL)
    models = CodeCompModelRoutesConfig(encoder=encoder)
    fields: dict[str, Any] = {
        "mode": mode,
        "models": models,
    }
    if mode is CodeCompMode.DIRECT:
        fields["direct"] = DirectModeSettings()
    elif mode is CodeCompMode.ENCDEC:
        fields["encdec"] = EncDecModeSettings(
            blend_config=CODE_COMP_DEFAULT_BLEND_CONFIG
        )
    else:
        fields["mutant"] = MutantModeSettings(
            artifact_dir=Path(cast(str, artifact_dir))
        )
    config = CodeCompExperimentConfig.model_validate(fields)
    update: dict[str, Any] = {}
    legacy_pool = {
        key: overrides.pop(key)
        for key in ("tasks", "snapshot_path", "limit")
        if key in overrides
    }
    if legacy_pool:
        update["pool"] = config.pool.model_copy(update=legacy_pool)
    legacy_split = {
        key: overrides.pop(key)
        for key in (
            "internal_n",
            "official_n",
            "split_manifest",
            "exclude_task_ids",
        )
        if key in overrides
    }
    if legacy_split:
        update["split"] = config.split.model_copy(update=legacy_split)
    legacy_sampling = {
        key: overrides.pop(key)
        for key in ("num_samples", "completeness", "max_skip_fraction")
        if key in overrides
    }
    if legacy_sampling:
        update["sampling"] = config.sampling.model_copy(update=legacy_sampling)
    if mode is CodeCompMode.DIRECT:
        direct_flat = {
            key: overrides.pop(key)
            for key in ("model", "input_arm", "rename_token")
            if key in overrides
        }
        if direct_flat:
            assert config.direct is not None
            update["direct"] = config.direct.model_copy(update=direct_flat)
    elif mode is CodeCompMode.ENCDEC:
        if "provider_call_config" in overrides:
            update["models"] = CodeCompModelRoutesConfig(
                encoder=CodeCompModelRouteConfig(
                    provider_call_config=overrides.pop("provider_call_config")
                )
            )
        encdec_flat = {
            key: overrides.pop(key)
            for key in ("budget_ratio", "blend_config")
            if key in overrides
        }
        if encdec_flat:
            assert config.encdec is not None
            update["encdec"] = config.encdec.model_copy(update=encdec_flat)
    else:
        if "provider_call_config" in overrides:
            update["models"] = CodeCompModelRoutesConfig(
                encoder=CodeCompModelRouteConfig(
                    provider_call_config=overrides.pop("provider_call_config")
                )
            )
        mutant_flat = {
            key: overrides.pop(key)
            for key in ("budget_ratio", "blend_config", "exclude_mutant_ids")
            if key in overrides
        }
        if mutant_flat:
            assert config.mutant is not None
            update["mutant"] = config.mutant.model_copy(update=mutant_flat)
    nested_keys = (
        "pool",
        "split",
        "sampling",
        "compression",
        "models",
        "direct",
        "encdec",
        "mutant",
    )
    nested_update: dict[str, Any] = {}
    for key in nested_keys:
        if key not in overrides:
            continue
        value = overrides.pop(key)
        if isinstance(value, dict):
            nested_update[key] = getattr(config, key).model_copy(update=value)
        else:
            nested_update[key] = value
    update.update(nested_update)
    if overrides:
        raise TypeError(f"unexpected keyword arguments: {sorted(overrides)}")
    if update:
        config = config.model_copy(update=update)
    return config


__all__ = [
    "CODE_COMP_EXPERIMENT_CONFIG_SCHEMA",
    "CODE_COMP_EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "CodeCompExperimentConfig",
    "CodeCompModelRouteConfig",
    "CodeCompModelRoutesConfig",
    "CodeCompPoolConfig",
    "CodeCompSamplingConfig",
    "CodeCompSplitConfig",
    "CompressionOperatorConfig",
    "DirectModeSettings",
    "EncDecModeSettings",
    "MutantModeSettings",
    "default_code_comp_config",
]
