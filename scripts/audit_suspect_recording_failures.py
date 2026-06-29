from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import psycopg
import typer
from pydantic import BaseModel, ConfigDict, StrictStr

from dr_dspy.dbos_runtime import resolve_database_url
from dr_dspy.prediction_status import GenerationStatus
from dr_dspy.runtime import load_env_file

DATABASE_URL_ENV = "DATABASE_URL"
DIRECT_PREDICTION_TABLE = "dr_dspy_eval_predictions"
ENCDEC_PREDICTION_TABLE = "dr_dspy_encdec_eval_predictions"
AUDIT_FLAG_KEY = "suspect_silent_recording_failure"

app = typer.Typer(no_args_is_help=True)


class AuditRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    tier: StrictStr


class TableAuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: StrictStr
    high_tier_sql: StrictStr
    medium_tier_sql: StrictStr
    low_tier_sql: StrictStr


def _experiment_clause(experiment_name: str | None) -> tuple[str, list[str]]:
    if experiment_name is None:
        return "", []
    return "AND experiment_name = %s", [experiment_name]


DIRECT_AUDIT = TableAuditConfig(
    table_name=DIRECT_PREDICTION_TABLE,
    high_tier_sql="""
        SELECT prediction_id, experiment_name
        FROM dr_dspy_eval_predictions
        WHERE
            generation_status = %s
            AND raw_generation IS NOT NULL
            AND btrim(raw_generation) <> ''
            AND response_metadata = '{}'::jsonb
            AND usage_metadata = '{}'::jsonb
            {experiment_clause}
    """,
    medium_tier_sql="""
        SELECT prediction_id, experiment_name
        FROM dr_dspy_eval_predictions
        WHERE
            generation_status = %s
            AND raw_generation IS NOT NULL
            AND btrim(raw_generation) <> ''
            AND response_metadata = '{}'::jsonb
            AND usage_metadata = '{}'::jsonb
            AND provider_cost IS NULL
            {experiment_clause}
    """,
    low_tier_sql="""
        SELECT prediction_id, experiment_name
        FROM dr_dspy_eval_predictions
        WHERE
            generation_status = %s
            AND raw_generation IS NOT NULL
            AND btrim(raw_generation) <> ''
            AND response_metadata = '{}'::jsonb
            AND usage_metadata = '{}'::jsonb
            AND (
                score = 0
                OR extraction_error IS NOT NULL
                OR evaluation_status_counts ? 'error'
            )
            {experiment_clause}
    """,
)

ENCDEC_AUDIT = TableAuditConfig(
    table_name=ENCDEC_PREDICTION_TABLE,
    high_tier_sql="""
        SELECT prediction_id, experiment_name
        FROM dr_dspy_encdec_eval_predictions
        WHERE
            generation_status = %s
            AND COALESCE(
                NULLIF(btrim(decoded_generation), ''),
                NULLIF(btrim(raw_generation), '')
            ) IS NOT NULL
            AND encoder_response_metadata = '{}'::jsonb
            AND decoder_response_metadata = '{}'::jsonb
            AND encoder_usage_metadata = '{}'::jsonb
            AND decoder_usage_metadata = '{}'::jsonb
            {experiment_clause}
    """,
    medium_tier_sql="""
        SELECT prediction_id, experiment_name
        FROM dr_dspy_encdec_eval_predictions
        WHERE
            generation_status = %s
            AND COALESCE(
                NULLIF(btrim(decoded_generation), ''),
                NULLIF(btrim(raw_generation), '')
            ) IS NOT NULL
            AND encoder_response_metadata = '{}'::jsonb
            AND decoder_response_metadata = '{}'::jsonb
            AND encoder_usage_metadata = '{}'::jsonb
            AND decoder_usage_metadata = '{}'::jsonb
            AND provider_cost IS NULL
            {experiment_clause}
    """,
    low_tier_sql="""
        SELECT prediction_id, experiment_name
        FROM dr_dspy_encdec_eval_predictions
        WHERE
            generation_status = %s
            AND COALESCE(
                NULLIF(btrim(decoded_generation), ''),
                NULLIF(btrim(raw_generation), '')
            ) IS NOT NULL
            AND encoder_response_metadata = '{}'::jsonb
            AND decoder_response_metadata = '{}'::jsonb
            AND encoder_usage_metadata = '{}'::jsonb
            AND decoder_usage_metadata = '{}'::jsonb
            AND (
                score = 0
                OR extraction_error IS NOT NULL
                OR evaluation_status_counts ? 'error'
            )
            {experiment_clause}
    """,
)

TABLE_CONFIGS = {
    DIRECT_PREDICTION_TABLE: DIRECT_AUDIT,
    ENCDEC_PREDICTION_TABLE: ENCDEC_AUDIT,
}


def fetch_tier_rows(
    conn: psycopg.Connection[object],
    *,
    config: TableAuditConfig,
    tier: str,
    experiment_name: str | None,
) -> list[AuditRow]:
    experiment_clause, experiment_params = _experiment_clause(experiment_name)
    query_template = {
        "high": config.high_tier_sql,
        "medium": config.medium_tier_sql,
        "low": config.low_tier_sql,
    }[tier]
    query = query_template.format(experiment_clause=experiment_clause)
    params = [GenerationStatus.GENERATED.value, *experiment_params]
    with conn.cursor() as cur:
        cur.execute(cast(Any, query), params)
        rows = cur.fetchall()
    return [
        AuditRow(
            prediction_id=str(cast(tuple[object, ...], row)[0]),
            experiment_name=str(cast(tuple[object, ...], row)[1]),
            tier=tier,
        )
        for row in rows
    ]


def apply_audit_flags(
    conn: psycopg.Connection[object],
    *,
    table_name: str,
    rows: Sequence[AuditRow],
    tier: str,
) -> int:
    if not rows:
        return 0
    audited_at = datetime.now(tz=UTC).isoformat()
    flag_payload = json.dumps(
        {
            AUDIT_FLAG_KEY: {
                "tier": tier,
                "audited_at": audited_at,
                "heuristic": "empty_metadata_generated",
            }
        }
    )
    prediction_ids = [row.prediction_id for row in rows]
    query = f"""
        UPDATE {table_name}
        SET
            data_quality_flags = data_quality_flags || %s::jsonb,
            updated_at = now()
        WHERE prediction_id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(cast(Any, query), (flag_payload, prediction_ids))
        return cur.rowcount if cur.rowcount is not None else 0


@app.command()
def main(
    experiment_name: Annotated[
        str | None,
        typer.Option(
            "--experiment-name",
            help="Limit audit to one experiment.",
        ),
    ] = None,
    table: Annotated[
        str,
        typer.Option(
            "--table",
            help="Prediction table to audit.",
        ),
    ] = "all",
    database_url: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
    apply: Annotated[bool, typer.Option("--apply")] = False,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            help="Sample ids to print per tier.",
        ),
    ] = 10,
) -> None:
    if env_file is None:
        load_env_file()
    else:
        load_env_file(env_file)
    resolved_database_url = resolve_database_url(
        database_url,
        database_url_env=DATABASE_URL_ENV,
    )

    if table == "all":
        configs = list(TABLE_CONFIGS.values())
    else:
        if table not in TABLE_CONFIGS:
            raise typer.BadParameter(
                f"unsupported table {table!r}; "
                f"expected one of {sorted(TABLE_CONFIGS)} or 'all'"
            )
        configs = [TABLE_CONFIGS[table]]

    with psycopg.connect(resolved_database_url) as conn:
        for config in configs:
            typer.echo(f"table={config.table_name}")
            for tier in ("high", "medium", "low"):
                rows = fetch_tier_rows(
                    conn,
                    config=config,
                    tier=tier,
                    experiment_name=experiment_name,
                )
                typer.echo(f"  {tier}: {len(rows)} row(s)")
                for row in rows[:sample_size]:
                    typer.echo(
                        f"    prediction_id={row.prediction_id} "
                        f"experiment={row.experiment_name}"
                    )
                if apply and rows:
                    updated = apply_audit_flags(
                        conn,
                        table_name=config.table_name,
                        rows=rows,
                        tier=tier,
                    )
                    typer.echo(f"  {tier}: tagged {updated} row(s)")
        if apply:
            conn.commit()
            typer.echo("committed audit flags")
        else:
            typer.echo("dry run only; rerun with --apply to tag rows")


if __name__ == "__main__":
    app()
