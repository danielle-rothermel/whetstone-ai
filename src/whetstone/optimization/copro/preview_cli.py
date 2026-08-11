from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dr_exec import (
    DirectoryRunStore,
    IsolatedHostPythonRuntime,
    ProcessExecutor,
)

from whetstone.optimization.codex.proposer import (
    CodexCliProposerConfig,
    CodexCliProposerTransport,
)
from whetstone.optimization.copro.code_comp.dry_run import (
    Ed1CoproPreviewTask,
    Ed1CoproSweepRanges,
    run_ed1_copro_codex_preview,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask Codex CLI for ED1 COPRO seed mutations and print the "
            "proposal-only transcript. No candidate evaluation is run."
        )
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--input-code-file", type=Path, required=True)
    parser.add_argument("--budget-ratio", type=float)
    parser.add_argument("--breadth", type=int, default=3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--model", default="")
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path.home() / ".cache/whetstone/copro-proposer-runs",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    records = args.records.resolve()
    records.mkdir(parents=True, exist_ok=True)
    executor = ProcessExecutor(
        runtime=IsolatedHostPythonRuntime(Path(sys.executable)),
        run_store=DirectoryRunStore(root=records),
    )
    transport = CodexCliProposerTransport(executor=executor)
    run_ed1_copro_codex_preview(
        sweep=Ed1CoproSweepRanges(
            budget_ratios=(args.budget_ratio,),
            breadths=(args.breadth,),
            depths=(args.depth,),
        ),
        preview_task=Ed1CoproPreviewTask(
            task_id=args.task_id,
            input_code=args.input_code_file.read_text(encoding="utf-8"),
        ),
        proposer_config=CodexCliProposerConfig(
            codex_binary=args.codex_binary,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        ),
        transport=transport,
        log=print,
    )


if __name__ == "__main__":
    main()
