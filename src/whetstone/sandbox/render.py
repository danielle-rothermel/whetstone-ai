from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from whetstone.sandbox.transcript import SandboxCoproSeedTranscript
from whetstone.sandbox.gepa_step import GepaStepPreview
from whetstone.sandbox.graph_step import GraphRunPreview
from whetstone.sandbox.miprov2_step import Miprov2PlanPreview

__all__ = [
    "render_copro_preview",
    "render_gepa_preview",
    "render_graph_preview",
    "render_miprov2_preview",
]


def render_copro_preview(
    console: Console,
    transcript: SandboxCoproSeedTranscript,
) -> None:
    plan = Table.grid(padding=(0, 2))
    plan.add_column(style="bold cyan")
    plan.add_column()
    plan.add_row("Breadth", str(transcript.breadth))
    plan.add_row("Depth", str(transcript.depth))
    plan.add_row("Round", str(transcript.round_plan.iteration))
    plan.add_row("Mode", transcript.round_plan.proposal_mode)
    plan.add_row("Proposal count", str(transcript.round_plan.proposal_count))
    console.print(Rule("[bold]COPRO seed preview"))
    console.print(
        Panel(
            Text(transcript.task_prompt),
            title="Baseline instruction",
            border_style="blue",
        )
    )
    console.print(Panel(plan, title="Round plan", border_style="cyan"))

    call = Table.grid(padding=(0, 2))
    call.add_column(style="bold green", no_wrap=True)
    call.add_column(overflow="fold")
    call.add_row("Mutation field", transcript.proposal_call.mutation_field)
    call.add_row("Base template", transcript.proposal_call.base_template)
    call.add_row("Prompt", transcript.proposal_call.prompt)
    console.print(Panel(call, title="Proposal call", border_style="green"))

    drafts = Table("#", "Template", "Failed", box=None)
    for draft in transcript.drafts:
        drafts.add_row(
            str(draft.ordinal),
            draft.template,
            "yes" if draft.failed else "no",
        )
    console.print(Panel(drafts, title="Drafts", border_style="magenta"))

    mutations = Table("#", "Candidate", "Disposition", "Template", box=None)
    for mutation in transcript.mutations:
        mutations.add_row(
            str(mutation.ordinal),
            mutation.candidate_id,
            mutation.disposition,
            mutation.template,
        )
    console.print(Panel(mutations, title="Mutations", border_style="yellow"))


def render_miprov2_preview(console: Console, preview: Miprov2PlanPreview) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Round", str(preview.round_index))
    table.add_row("Surface", preview.surface)
    table.add_row("Proposal mode", preview.proposal_mode)
    table.add_row("Mutation field", preview.mutation_field)
    table.add_row("Components", str(preview.component_count))
    console.print(Rule("[bold]MIPROv2 plan preview"))
    console.print(Panel(Text(preview.base_template), title="Base template"))
    console.print(Panel(Text(preview.message), title="Boundary", border_style="yellow"))


def render_gepa_preview(console: Console, preview: GepaStepPreview) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Component", preview.selected_component)
    table.add_row("Mutation field", preview.mutation_field)
    table.add_row("Show intent", "yes" if preview.show_intent else "no")
    console.print(Rule("[bold]GEPA step preview"))
    console.print(Panel(Text(preview.current_text), title="Current component text"))
    console.print(
        Panel(Text(preview.evaluation_intent_boundary), title="Intent boundary")
    )


def render_graph_preview(console: Console, preview: GraphRunPreview) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Graph hash", preview.graph_hash[:16] + "…")
    table.add_row("LLM node", preview.llm_node_id)
    table.add_row("Eval node", preview.eval_node_id)
    table.add_row("Score", "n/a" if preview.score is None else f"{preview.score:.4f}")
    console.print(Rule("[bold]Toy graph run"))
    console.print(Panel(Text(preview.prompt), title="Prompt"))
    console.print(Panel(Text(preview.generation), title="Generation"))
    console.print(Panel(table, title="Outcome", border_style="green"))
