from __future__ import annotations


def validate_sql_identifier(identifier: str) -> None:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"unsupported SQL identifier: {identifier}")


def repair_plan_line(
    *,
    experiment_name: str,
    gen_stranded: int,
    gen_errors: int,
    gen_recoverable_errors: int,
    gen_excluded_errors: int,
    score_pending: int,
    score_stranded: int,
    score_errors: int,
    score_recoverable_errors: int,
    score_excluded_errors: int,
    apply: bool,
) -> str:
    mode = "apply" if apply else "dry-run"
    return (
        f"{'Repair Plan':<14} | "
        f"gen_stranded={gen_stranded:>5} | "
        f"gen_retry={gen_errors:>5} "
        f"(rec={gen_recoverable_errors}, "
        f"skip={gen_excluded_errors}) | "
        f"score_pending={score_pending:>5} | "
        f"score_stranded={score_stranded:>5} | "
        f"score_retry={score_errors:>5} "
        f"(rec={score_recoverable_errors}, "
        f"skip={score_excluded_errors}) | "
        f"mode={mode} | "
        f"experiment={experiment_name}"
    )


def repair_plan_style(
    *,
    apply: bool,
    gen_stranded: int,
    gen_errors: int,
    gen_recoverable_errors: int,
    gen_excluded_errors: int,
    score_pending: int,
    score_stranded: int,
    score_errors: int,
    score_recoverable_errors: int,
    score_excluded_errors: int,
) -> str:
    if apply:
        return "green"
    if (
        gen_stranded
        or gen_errors
        or gen_recoverable_errors
        or gen_excluded_errors
        or score_pending
        or score_stranded
        or score_errors
        or score_recoverable_errors
        or score_excluded_errors
    ):
        return "cyan"
    return "yellow"
