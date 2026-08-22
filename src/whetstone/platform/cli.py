from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

import typer
from sqlalchemy import create_engine

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.coordination.harness_run_controller import load_launch
from whetstone.coordination.runtime_bootstrap import build_runtime
from whetstone.core.identity import compute_identity_hash, store_config_resolver
from whetstone.core.leasing import EffectLeaseAuthority
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
    ProviderProposerTransport,
    build_inline_proposal_executor,
    prompt_adapter_identity_hash,
)
from whetstone.platform.contracts import load_run_manifest, load_run_result
from whetstone.platform.submit import OptimRunMemberSpec, submit_optim_run
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.provider.policy import ProviderExecutionPolicy

app = typer.Typer(add_completion=False, no_args_is_help=True)

ADAPTER_COPRO = "copro"
ADAPTER_GEPA = "gepa"
PROPOSER_PROVIDER = "provider"
PROPOSER_FAKE = "fake"


def _require_database_url(database_url: str | None) -> str:
    resolved = database_url or os.environ.get("WHETSTONE_DATABASE_URL")
    if not resolved:
        raise typer.BadParameter(
            "pass --database-url or set WHETSTONE_DATABASE_URL"
        )
    return resolved


def _stable_owner_id(*, application_version: str, executor_id: str) -> str:
    return f"{application_version}:{executor_id}"


def _live_transport_call(execution_policy: ProviderExecutionPolicy):
    from dr_providers import HttpProvider

    return HttpProvider(policy=execution_policy.transport_policy).invoke


def _copro_adapter_from_control(
    control,
    engine,
    *,
    store,
    proposer: str,
    execution_policy: ProviderExecutionPolicy,
) -> CoproAdapter:
    prompt_adapter = PlainPromptAdapter()
    adapter_hash = prompt_adapter_identity_hash(prompt_adapter)
    if control.prompt_adapter_identity_hash != adapter_hash:
        raise typer.BadParameter(
            "launch COPRO control prompt adapter does not match "
            "the CLI PlainPromptAdapter"
        )
    if (
        control.expected_reward_policy_hash
        != engine.reward_policy_identity_hash()
    ):
        raise typer.BadParameter(
            "launch control reward policy does not match the rebuilt engine"
        )
    if control.eval_config_ref != engine.eval_config_ref:
        raise typer.BadParameter(
            "launch control eval_config_ref does not match the rebuilt engine"
        )
    if proposer == PROPOSER_FAKE:
        transport: (
            FakeProposerTransport | ProviderProposerTransport
        ) = FakeProposerTransport(
            {},
            default=(
                "Reply briefly to: {prompt} with a concise greeting.",
                "Answer {prompt} in one short friendly sentence.",
            ),
            execution_policy_hash=execution_policy.identity_hash,
            prompt_adapter_identity_hash=adapter_hash,
        )
    elif proposer == PROPOSER_PROVIDER:
        if not isinstance(control.prompt_model, ProposerConfig):
            raise typer.BadParameter(
                "provider proposer requires a stored ProviderCallConfig"
            )
        from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA, ProviderCallConfig

        transport = ProviderProposerTransport(
            resolve_provider_call_config=store_config_resolver(
                store,
                PROVIDER_CALL_CONFIG_SCHEMA,
                ProviderCallConfig,
            ),
            transport=_live_transport_call(execution_policy),
            execution_policy=execution_policy,
            prompt_adapter=prompt_adapter,
        )
    else:
        raise typer.BadParameter("proposer must be provider or fake")
    return CoproAdapter(
        control=control,
        transport=transport,
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema="whetstone.inline_proposal_executor",
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )


def _receipt_payload(receipt) -> dict[str, object]:
    return {
        "run_key": receipt.run_key.value,
        "membership_digest": receipt.membership_digest,
        "registered_member_count": receipt.registered_member_count,
        "created_work_count": receipt.created_work_count,
        "reused_work_count": receipt.reused_work_count,
        "registration_closed_at": receipt.registration_closed_at.isoformat(),
    }


@app.command("run")
def run_command(
    run_id: Annotated[str, typer.Option("--run-id", help="Bound launch run id")],
    store_path: Annotated[str, typer.Option("--store-path", help="SQLite store path")],
    campaign_key: Annotated[str, typer.Option("--campaign-key")],
    run_key: Annotated[str, typer.Option("--run-key")],
    adapter: Annotated[
        str,
        typer.Option("--adapter", help="Adapter to reconstruct from the launch"),
    ] = ADAPTER_COPRO,
    proposer: Annotated[
        Literal["provider", "fake"],
        typer.Option(
            "--proposer",
            help="COPRO proposer transport: live provider or scripted fake",
        ),
    ] = PROPOSER_PROVIDER,
    work_key: Annotated[str | None, typer.Option("--work-key")] = None,
    database_url: Annotated[
        str | None,
        typer.Option("--database-url", help="Postgres URL; or WHETSTONE_DATABASE_URL"),
    ] = None,
    dispatch_mode: Annotated[
        EvalDispatchMode,
        typer.Option("--dispatch-mode"),
    ] = EvalDispatchMode.INLINE,
    application_version: Annotated[
        str,
        typer.Option("--application-version"),
    ] = "whetstone-optim",
    executor_id: Annotated[str, typer.Option("--executor-id")] = "whetstone-optim-cli",
    owner_id: Annotated[
        str | None,
        typer.Option(
            "--owner-id",
            help=(
                "Stable controller owner id; defaults to "
                "application-version:executor-id"
            ),
        ),
    ] = None,
    execution_config_ref: Annotated[
        str | None,
        typer.Option("--execution-config-ref"),
    ] = None,
) -> None:
    """Submit a bound optimization launch through the platform pipeline."""
    from dr_store.sync import open_sqlite

    from whetstone.platform.deploy import (
        await_run_completion,
        deploy_platform,
        drive_until_quiescent,
    )

    if adapter not in {ADAPTER_COPRO, ADAPTER_GEPA}:
        raise typer.BadParameter("adapter must be copro or gepa")
    resolved_database_url = _require_database_url(database_url)
    resolved_work_key = work_key or run_id
    resolved_exec_ref = execution_config_ref or f"exec-config-{uuid4().hex[:10]}"
    ledger = create_engine(resolved_database_url)
    try:
        with open_sqlite(store_path) as store:
            launch = load_launch(store, run_id)
            if launch.run.adapter_key != adapter:
                raise typer.BadParameter(
                    f"launch adapter {launch.run.adapter_key!r} does not "
                    f"match --adapter {adapter!r}"
                )
            if adapter == ADAPTER_GEPA:
                raise typer.BadParameter(
                    "GEPA reconstruction from a stored launch is not "
                    "supported; pass a GEPA adapter to build_runtime"
                )
            if launch.control is None:
                raise typer.BadParameter("launch is missing optimizer control")
            runtime_config = ReferenceEvalRuntimeConfig()
            engine = runtime_config.build_engine(
                store,
                mutation_field=launch.run.mutation_field,
                render_contract=launch.run.template_render_contract,
            )
            copro_adapter = _copro_adapter_from_control(
                launch.control,
                engine,
                store=store,
                proposer=proposer,
                execution_policy=runtime_config.execution_policy,
            )
            resolved_owner_id = owner_id or _stable_owner_id(
                application_version=application_version,
                executor_id=executor_id,
            )
            runtime = build_runtime(
                store=store,
                engine=engine,
                adapter_registry=MappingAdapterRegistry(
                    {COPRO_ADAPTER_KEY: copro_adapter}
                ),
                effect_authority=EffectLeaseAuthority.memory(),
                ledger_engine=ledger,
                platform=True,
                owner_id=resolved_owner_id,
            )
            with runtime:
                now = datetime.now(UTC)
                deployment = deploy_platform(
                    runtime=runtime,
                    engine=ledger,
                    database_url=resolved_database_url,
                    application_version=application_version,
                    executor_id=executor_id,
                    now=now,
                    app_name="whetstone-optim",
                    sweep_cron=None,
                )
                try:
                    receipt = submit_optim_run(
                        runtime=runtime,
                        registry=deployment.registry,
                        engine=ledger,
                        campaign_key=campaign_key,
                        run_key=run_key,
                        members=(
                            OptimRunMemberSpec(
                                work_key=resolved_work_key,
                                launch=launch,
                            ),
                        ),
                        controller_identity_hash=runtime.controller.runtime_hash,
                        execution_config_reference=resolved_exec_ref,
                        dispatch_mode=dispatch_mode,
                    )
                    typer.echo(json.dumps(_receipt_payload(receipt), indent=2))
                    drive_until_quiescent(
                        engine=ledger,
                        registry=deployment.registry,
                        registration=deployment.registration,
                        now=now,
                        run_key=run_key,
                    )
                    await_run_completion(
                        run_key=run_key,
                        engine=ledger,
                        registration=deployment.registration,
                        registry=deployment.registry,
                        now=now,
                    )
                finally:
                    deployment.shutdown()
    finally:
        ledger.dispose()


@app.command("status")
def status_command(
    run_key: Annotated[str, typer.Option("--run-key")],
    store_path: Annotated[str, typer.Option("--store-path")],
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
) -> None:
    """Print the platform run manifest and release state."""
    from dr_store.sync import open_sqlite

    from whetstone.platform.deploy import load_pipeline_run

    resolved_database_url = _require_database_url(database_url)
    ledger = create_engine(resolved_database_url)
    try:
        record = load_pipeline_run(ledger, run_key=run_key)
        if record.manifest_reference is None:
            raise typer.Exit(code=1)
        with open_sqlite(store_path) as store:
            manifest = load_run_manifest(store, record.manifest_reference)
        typer.echo(
            json.dumps(
                {
                    "run_key": record.run_key.value,
                    "campaign_key": record.campaign_key.value,
                    "released": record.released_at is not None,
                    "released_at": (
                        None
                        if record.released_at is None
                        else record.released_at.isoformat()
                    ),
                    "membership_digest": record.membership_digest,
                    "members": [
                        {
                            "work_key": member.work_key,
                            "run_id": member.run_id,
                        }
                        for member in manifest.members
                    ],
                },
                indent=2,
            )
        )
    finally:
        ledger.dispose()


@app.command("result")
def result_command(
    run_key: Annotated[str, typer.Option("--run-key")],
    store_path: Annotated[str, typer.Option("--store-path")],
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
) -> None:
    """Print the OptimPlatformRunResult for a completed platform run."""
    from dr_platform.completion.execution import inspect_run_completion
    from dr_store.sync import open_sqlite

    resolved_database_url = _require_database_url(database_url)
    ledger = create_engine(resolved_database_url)
    try:
        execution = inspect_run_completion(run_key, engine=ledger)
        if execution.output_reference is None:
            raise typer.BadParameter(
                f"run {run_key!r} has no completion output yet"
            )
        with open_sqlite(store_path) as store:
            run_result = load_run_result(store, execution.output_reference)
        typer.echo(run_result.model_dump_json(indent=2))
    finally:
        ledger.dispose()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
