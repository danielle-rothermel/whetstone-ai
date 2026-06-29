from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import psycopg
import typer
from pydantic import BaseModel, ConfigDict, StrictStr

from dr_dspy.eval_failures import FailureClass
from dr_dspy.harness.dbos import resolve_database_url
from dr_dspy.harness.status import GenerationStatus
from dr_dspy.runtime import load_env_file

DATABASE_URL_ENV = "DATABASE_URL"
PREDICTION_TABLE_NAME = "dr_dspy_encdec_eval_predictions"
HTTP_STATUS_ERROR_TYPE = "httpx.HTTPStatusError"
UNKNOWN_FAILURE_CLASS = "unknown"
HTTP_429_MESSAGE_FRAGMENT = "429 Too Many Requests"
HTTP_504_MESSAGE_FRAGMENT = "504 Gateway Timeout"


class ReclassificationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_fragment: StrictStr
    failure_class: FailureClass


RECLASSIFICATION_RULES = (
    ReclassificationRule(
        message_fragment=HTTP_429_MESSAGE_FRAGMENT,
        failure_class=FailureClass.RATE_LIMITED,
    ),
    ReclassificationRule(
        message_fragment=HTTP_504_MESSAGE_FRAGMENT,
        failure_class=FailureClass.TRANSIENT,
    ),
)

app = typer.Typer(no_args_is_help=True)


def matching_row_count(
    conn: psycopg.Connection[object],
    *,
    experiment_name: str,
    rule: ReclassificationRule,
) -> int:
    query = f"""
        SELECT COUNT(*)
        FROM {PREDICTION_TABLE_NAME}
        WHERE
            experiment_name = %s
            AND generation_status = %s
            AND generation_failure_class = %s
            AND generation_underlying_exception_type = %s
            AND generation_exception_message LIKE %s
    """
    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                experiment_name,
                GenerationStatus.ERROR.value,
                UNKNOWN_FAILURE_CLASS,
                HTTP_STATUS_ERROR_TYPE,
                f"%{rule.message_fragment}%",
            ),
        )
        row = cur.fetchone()
    if row is None:
        return 0
    count_row = cast(tuple[int], row)
    return int(count_row[0])


def reclassify_matching_rows(
    conn: psycopg.Connection[object],
    *,
    experiment_name: str,
    rule: ReclassificationRule,
) -> int:
    query = f"""
        UPDATE {PREDICTION_TABLE_NAME}
        SET
            generation_status = %s,
            generation_failure_class = %s,
            generation_error = %s
                || ': '
                || COALESCE(generation_underlying_exception_type, '')
                || ': '
                || COALESCE(generation_exception_message, ''),
            updated_at = now()
        WHERE
            experiment_name = %s
            AND generation_status = %s
            AND generation_failure_class = %s
            AND generation_underlying_exception_type = %s
            AND generation_exception_message LIKE %s
    """
    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                GenerationStatus.RECOVERABLE_ERROR.value,
                rule.failure_class.value,
                rule.failure_class.value,
                experiment_name,
                GenerationStatus.ERROR.value,
                UNKNOWN_FAILURE_CLASS,
                HTTP_STATUS_ERROR_TYPE,
                f"%{rule.message_fragment}%",
            ),
        )
        return cur.rowcount if cur.rowcount is not None else 0


@app.command()
def main(
    experiment_name: Annotated[
        str, typer.Option("--experiment-name", help="Experiment to update.")
    ],
    database_url: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
    apply: Annotated[bool, typer.Option("--apply")] = False,
) -> None:
    if env_file is None:
        load_env_file()
    else:
        load_env_file(env_file)
    resolved_database_url = resolve_database_url(
        database_url,
        database_url_env=DATABASE_URL_ENV,
    )

    with psycopg.connect(resolved_database_url) as conn:
        for rule in RECLASSIFICATION_RULES:
            count = matching_row_count(
                conn,
                experiment_name=experiment_name,
                rule=rule,
            )
            typer.echo(
                f"{rule.message_fragment}: {count} row(s) -> "
                f"{rule.failure_class.value}"
            )
            if apply and count:
                updated = reclassify_matching_rows(
                    conn,
                    experiment_name=experiment_name,
                    rule=rule,
                )
                typer.echo(f"updated {updated} row(s)")
        if apply:
            conn.commit()
            typer.echo("committed reclassification")
        else:
            typer.echo("dry run only; rerun with --apply to update rows")


if __name__ == "__main__":
    app()
