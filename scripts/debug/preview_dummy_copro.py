#!/usr/bin/env python3
"""Print a deterministic, human-readable ED1 COPRO seed-round preview."""

import argparse

from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from whetstone.optimization.copro.ed1_dry_run import (
    DummyCoproProposerConfig,
    Ed1CoproDryRunTranscript,
    Ed1CoproPreviewTask,
    Ed1CoproProposalCall,
    Ed1CoproSweepRanges,
    Ed1CoproSweepTranscript,
    Ed1PromptPreview,
    run_ed1_copro_dry_run,
)


def _axis(values: tuple[object, ...]) -> str:
    return ", ".join(
        "none" if value is None else str(value) for value in values
    )


def render_inputs(
    console: Console,
    *,
    sweep: Ed1CoproSweepRanges,
    preview_task: Ed1CoproPreviewTask,
    dummy_proposer: DummyCoproProposerConfig,
) -> None:
    """Render the inputs that determine this deterministic preview."""

    sweep_table = Table.grid(padding=(0, 2))
    sweep_table.add_column(style="bold cyan")
    sweep_table.add_column()
    sweep_table.add_row("Budget ratios", _axis(sweep.budget_ratios))
    sweep_table.add_row("Breadths", _axis(sweep.breadths))
    sweep_table.add_row("Depths", _axis(sweep.depths))

    proposer_table = Table("#", "Instruction body", box=None)
    for ordinal, body in enumerate(dummy_proposer.bodies, start=1):
        proposer_table.add_row(str(ordinal), body)

    console.print(Rule("[bold]Dry-run inputs"))
    console.print(Panel(sweep_table, title="Sweep", border_style="cyan"))
    console.print(
        Panel(
            Syntax(preview_task.input_code, "python", word_wrap=True),
            title=f"Preview task · {preview_task.task_id}",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            proposer_table,
            title="Dummy proposer outputs",
            border_style="magenta",
        )
    )


def render_prompt(
    console: Console, prompt: Ed1PromptPreview, *, title: str
) -> None:
    """Render a prompt's literal mutation, template, fill, and final text."""

    contract = Table.grid(padding=(0, 2))
    contract.add_column(style="bold cyan", no_wrap=True)
    contract.add_column(overflow="fold")
    contract.add_row("Body literal", Text(repr(prompt.body_literal)))
    contract.add_row("Frame template", Text(repr(prompt.frame_template)))
    contract.add_row(
        "Fill",
        Pretty(prompt.fill.model_dump(mode="json"), expand_all=True),
    )

    rendered = Text(prompt.rendered_prompt)
    console.print(
        Panel(
            Group(
                contract, Rule("Model-visible prompt", style="dim"), rendered
            ),
            title=title,
            border_style="green",
        )
    )


def render_flow(console: Console) -> None:
    """Explain the dry-run control flow and its deliberate stopping point."""

    flow = Tree("[bold]run_ed1_copro_dry_run(...)")
    flow.add("Create the hand-engineered baseline candidate once")
    sweep_loop = flow.add("For each independent sweep point")
    sweep_loop.add("Construct CoproDriver from that point's breadth and depth")
    sweep_loop.add("Initialize fresh state from the shared baseline")
    sweep_loop.add("Advance the driver to produce the seed-round plan")
    sweep_loop.add("Bind the ED1 instruction-body proposal contract")
    sweep_loop.add("Build and record the exact ProposalRequest")
    sweep_loop.add("Call transport.draft(config, request, count)")
    sweep_loop.add("Record every returned ProposalDraft and its evidence")
    sweep_loop.add("Validate each body and construct candidate mutations")
    sweep_loop.add("Render each candidate with that point's budget framing")
    flow.add("Return one typed transcript containing every sweep point")

    console.print(
        Panel(
            Group(
                flow,
                Text(
                    "Stops before evaluation, result folding, winner "
                    "selection, or a later COPRO round.",
                    style="bold yellow",
                ),
            ),
            title="Actual dry-run flow",
            border_style="yellow",
        )
    )


def render_point_calls(
    console: Console,
    point: Ed1CoproSweepTranscript,
) -> None:
    """Show the significant calls and returns for one independent point."""

    plan = point.round_plan
    state = point.initial_state
    call = point.proposal_call
    candidate_ids = [
        mutation.candidate.record.candidate_id
        for mutation in point.candidate_mutations
    ]

    calls = Table("#", "Call", "Input", "Return", expand=True)
    calls.add_row(
        "1",
        "CoproDriver(config)",
        Pretty(point.settings.copro.model_dump(mode="json")),
        "Fresh lifecycle driver",
    )
    calls.add_row(
        "2",
        "driver.initial_state(baseline)",
        point.baseline_candidate.record.candidate_id,
        Pretty(
            {
                "completed_rounds": state.completed_rounds,
                "attempts": len(state.attempts),
                "total_calls": state.total_calls,
            }
        ),
    )
    calls.add_row(
        "3",
        "driver.advance(state)",
        "Fresh state",
        Pretty(plan.model_dump(mode="json"), expand_all=True),
    )
    calls.add_row(
        "4",
        "Build ProposalRequest",
        "Contract + base instruction + selected history",
        call.request.identity_hash(),
    )
    calls.add_row(
        "5",
        "transport.draft(config, request, count)",
        f"{call.proposer_kind}, count={call.requested_count}",
        f"{len(call.drafts)} recorded ProposalDrafts",
    )
    calls.add_row(
        "6",
        "Validate and construct candidates",
        f"{len(call.drafts)} returned bodies + baseline",
        Pretty(candidate_ids, expand_all=True),
    )
    console.print(
        Panel(
            calls, title="Calls for this sweep point", border_style="magenta"
        )
    )


def render_proposal_call(console: Console, call: Ed1CoproProposalCall) -> None:
    """Render the exact shared proposer contract, request, and response."""

    contract = call.instruction_contract
    contract_table = Table.grid(padding=(0, 2))
    contract_table.add_column(style="bold cyan", no_wrap=True)
    contract_table.add_column(overflow="fold")
    contract_table.add_row("Target", contract.target_name)
    contract_table.add_row("Budget mode", contract.budget_mode)
    contract_table.add_row("Task", contract.task_context)
    contract_table.add_row("Encoder frame", Text(repr(contract.encoder_frame)))
    contract_table.add_row(
        "Fixed decoder", Text(repr(contract.decoder_template))
    )
    contract_table.add_row("Output rule", contract.output_rule)
    console.print(
        Panel(
            contract_table,
            title="Identity-bound proposal contract",
            border_style="yellow",
        )
    )

    history = call.request.context["instruction_history"]
    assert isinstance(history, tuple)
    request_table = Table.grid(padding=(0, 2))
    request_table.add_column(style="bold cyan", no_wrap=True)
    request_table.add_column(overflow="fold")
    request_table.add_row("Proposer", call.proposer_kind)
    request_table.add_row(
        "Config identity", call.proposer_config_identity_hash
    )
    request_table.add_row(
        "Config", Pretty(call.proposer_config.to_json(), expand_all=True)
    )
    request_table.add_row("Mode", str(call.request.proposal_mode))
    request_table.add_row("Ordinal", str(call.request.request_ordinal))
    request_table.add_row("Base instruction", call.request.base_template)
    request_table.add_row("History entries", str(len(history)))
    request_table.add_row("Requested count", str(call.requested_count))
    request_table.add_row("Request identity", call.request.identity_hash())
    request_table.add_row(
        "Transport identity", call.transport_durability_identity_hash
    )
    console.print(
        Panel(
            request_table,
            title="Exact proposer call",
            border_style="magenta",
        )
    )

    proposal_prompt = call.request.context["proposal_prompt"]
    assert isinstance(proposal_prompt, str)
    console.print(
        Panel(
            Text(proposal_prompt),
            title="Prompt received by the proposer",
            border_style="blue",
        )
    )

    drafts = Table("#", "Status", "Instruction body", "Evidence", expand=True)
    for index, draft in enumerate(call.drafts, start=1):
        status = "failed" if draft.failed else "accepted"
        evidence = {
            "request": draft.request_evidence.to_json(),
            "response": draft.response_evidence.to_json(),
            "usage": draft.usage.to_json(),
            "cost": draft.cost,
        }
        drafts.add_row(
            str(index),
            status,
            draft.template or "—",
            Pretty(evidence, expand_all=True),
        )
    console.print(
        Panel(drafts, title="Recorded ProposalDrafts", border_style="green")
    )


def render_sweep_point(
    console: Console,
    point: Ed1CoproSweepTranscript,
    *,
    point_count: int,
) -> None:
    """Render one initialized COPRO lifecycle and its seed mutations."""

    settings = point.settings
    plan = point.round_plan
    console.print(
        Rule(f"[bold]Sweep point {settings.sweep_ordinal + 1}/{point_count}")
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Budget ratio", str(settings.budget_ratio))
    summary.add_row("Breadth", str(settings.copro.breadth))
    summary.add_row("Depth", str(settings.copro.depth))
    summary.add_row("Lifecycle round", str(plan.iteration))
    summary.add_row("Proposal mode", plan.proposal_mode)
    summary.add_row("Requested mutations", str(plan.proposal_count))
    summary.add_row(
        "Includes baseline",
        "yes" if plan.include_initial_candidate else "no",
    )
    console.print(Panel(summary, title="Lifecycle start", border_style="cyan"))
    render_point_calls(console, point)
    render_proposal_call(console, point.proposal_call)
    render_prompt(console, point.baseline_prompt, title="Baseline candidate")

    for ordinal, mutation in enumerate(point.candidate_mutations, start=1):
        render_prompt(
            console,
            mutation.prompt,
            title=(
                f"Mutation {ordinal} · "
                f"{mutation.candidate.record.candidate_id}"
            ),
        )


def render_transcript(
    console: Console, transcript: Ed1CoproDryRunTranscript
) -> None:
    """Render every initialized sweep point in a COPRO dry-run transcript."""

    console.print(Rule("[bold]Dry-run transcript"))
    render_flow(console)
    for point in transcript.points:
        render_sweep_point(
            console,
            point,
            point_count=len(transcript.points),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the raw transcript JSON instead of the Rich preview",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sweep = Ed1CoproSweepRanges(
        budget_ratios=(None, 0.5),
        breadths=(3,),
        depths=(1,),
    )
    preview_task = Ed1CoproPreviewTask(
        task_id="HumanEval/0",
        input_code="def add(a, b):\n    return a + b",
    )
    dummy_proposer = DummyCoproProposerConfig(
        bodies=(
            "Describe the function's behavior for a Python implementer",
            "Explain how to reconstruct an equivalent Python function",
        )
    )
    transcript = run_ed1_copro_dry_run(
        sweep=sweep,
        preview_task=preview_task,
        dummy_proposer=dummy_proposer,
    )
    if args.json:
        print(transcript.model_dump_json(indent=2))
        return

    console = Console()
    render_inputs(
        console,
        sweep=sweep,
        preview_task=preview_task,
        dummy_proposer=dummy_proposer,
    )
    render_transcript(console, transcript)


if __name__ == "__main__":
    main()
