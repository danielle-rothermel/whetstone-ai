from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from whetstone_envs.core import Instance, ProbePair

from whetstone.experiment.candidate import (
    TemplateRenderContract,
    TemplateRenderKind,
)


@dataclass(frozen=True, slots=True)
class ProbeSurface:
    """A mutable, serialization-stable naive/ceiling template pair + render.

    ``naive_template`` / ``ceiling_template`` are the two Mutation-Surface
    templates surfaced as Candidate payloads. ``render`` maps
    ``(template, instance)`` to a rendered prompt using only the template's
    *content* and the instance's public ``prompt_inputs`` -- never object
    identity and never gold. Because rendering is content-driven, a mutated or
    JSON-round-tripped template still renders.
    """

    naive_template: str
    ceiling_template: str
    template_render_contract: TemplateRenderContract
    render: Callable[[str, Instance], str]


def _render_with_contract(
    contract: TemplateRenderContract,
    values: Callable[[Instance], Mapping[str, object]],
) -> Callable[[str, Instance], str]:
    def render(template: str, instance: Instance) -> str:
        return contract.render(template, values(instance))

    return render


def _from_probe_pair(env_name: str, probes: ProbePair) -> ProbeSurface:
    """Wrap the content-driven ``ProbePair`` used by every bound environment
    except c19.
    """
    available_fields = {
        "c22": ("constraints_block",),
        "c22h": ("constraints_block",),
        "c11": ("input",),
        "c18": ("question", "query"),
        "c18h": ("question", "query"),
        "c23": ("demos_block", "query"),
    }[env_name]
    kind = (
        TemplateRenderKind.LITERAL_REPLACE_V1
        if env_name == "c11"
        else TemplateRenderKind.PYTHON_FORMAT_V1
    )
    contract = TemplateRenderContract(
        kind=kind,
        available_fields=available_fields,
    )
    return ProbeSurface(
        naive_template=probes.naive_template,
        ceiling_template=probes.ceiling_template,
        template_render_contract=contract,
        render=_render_with_contract(
            contract, lambda instance: dict(instance.prompt_inputs)
        ),
    )


# --- c19: replace sentinel-dispatch with real format templates ------------
#
# c19's own ``ProbePair`` stores sentinels and dispatches by identity; the
# fact-line varies by the public ``fact_type`` input. We rebuild the two
# templates as genuine ``str.format`` templates with a ``{fact_line}`` slot,
# import the env's exact head/fact-line text so the rendered bytes match the
# env renderer, and format against the public inputs only.

#: The ``str.format`` field marker for the per-fact-type question line. The
#: env heads use uppercase ``{GRID}`` / ``{COMMAND}`` placeholders that a
#: ``str.format`` call would misread as fields, so we translate those to the
#: real ``prompt_inputs`` keys (``{grid}`` / ``{command}``) when building the
#: adapter template.
_C19_FACT_LINE_SLOT = "{fact_line}"


def _c19_template(head: str) -> str:
    """Translate a c19 env head into a real ``str.format`` template.

    The env head carries ``{GRID}`` / ``{COMMAND}`` placeholders (substituted
    by ``str.replace`` in the env renderer) and no other braces. We map those
    to the public ``prompt_inputs`` keys and append the ``{fact_line}`` slot,
    yielding a template whose only fields are ``grid`` / ``command`` /
    ``fact_line`` -- all public.
    """
    body = head.replace("{GRID}", "{grid}").replace("{COMMAND}", "{command}")
    return body + _C19_FACT_LINE_SLOT


#: A stable substring present only in the c19 ceiling head (its rule
#: preamble). The single render callable is handed whichever template the
#: caller selected (or an edited descendant); it picks the ceiling vs naive
#: per-fact-type line table by this content marker. Content-driven, so it
#: survives a JSON round-trip and tolerates edits that keep the marker; an
#: edit that drops it falls back to the naive table (the safe floor).
_C19_CEILING_MARKER = "Follow these rules EXACTLY"


def _c19_values_for_template(
    template: str, instance: Instance
) -> Mapping[str, object]:
    from whetstone_envs.c19 import prompts as c19_prompts

    fact_lines = (
        dict(c19_prompts._CEILING_QUESTION_LINE)
        if _C19_CEILING_MARKER in template
        else dict(c19_prompts._NAIVE_FACT_LINE)
    )
    inputs = dict(instance.prompt_inputs)
    fact_type = inputs["fact_type"]
    if fact_type not in fact_lines:
        msg = f"no probe fact-line for fact type {fact_type!r}"
        raise KeyError(msg)
    return {
        "grid": inputs["grid"],
        "command": inputs["command"],
        "fact_line": fact_lines[fact_type],
    }


def _c19_surface() -> ProbeSurface:
    """The real, mutable, serialization-stable c19 probe surface.

    The env's own head/fact-line text is imported so the rendered bytes are
    byte-for-byte identical to the env's ``render_naive`` / ``render_ceiling``
    (pinned by a test), preserving oracle fidelity while the surfaced template
    is a genuine format template rather than an identity sentinel.
    """
    from whetstone_envs.c19 import prompts as c19_prompts

    contract = TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=("grid", "command", "fact_line"),
    )

    def render(template: str, instance: Instance) -> str:
        return contract.render(
            template, _c19_values_for_template(template, instance)
        )

    return ProbeSurface(
        naive_template=_c19_template(c19_prompts._NAIVE_HEAD),
        ceiling_template=_c19_template(c19_prompts._CEILING_HEAD),
        template_render_contract=contract,
        render=render,
    )


_C19_ENV = "c19"


def probe_surface(env_name: str, probes: ProbePair) -> ProbeSurface:
    """The adapter probe surface for ``env_name``.

    Returns a genuinely mutable, serialization-stable template pair + render.
    For c19 this is a real format-template surface (the env's identity-sentinel
    ``ProbePair`` is replaced); for every other env the env's own
    content-driven ``ProbePair`` is wrapped verbatim.
    """
    if env_name == _C19_ENV:
        return _c19_surface()
    return _from_probe_pair(env_name, probes)


__all__ = [
    "ProbeSurface",
    "probe_surface",
]
