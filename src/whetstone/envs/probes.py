"""The adapter-side probe surface: a genuinely mutable, serialization-stable
prompt template + a content-driven render for every env.

The Mutation Surface contract (``concrete-changes.html`` shared harness
expectation: "Mutation Surface = prompt template only") requires that the
``user_prompt_template`` an adapter surfaces as a Candidate payload be a *real*
template: an optimizer may rewrite it, it survives a Result-Store JSON
round-trip (``Candidate.model_validate_json``), and the resulting text renders
purely from the template *content* plus the task's public prompt inputs.

Four of the five envs already satisfy this: their ``ProbePair.render`` is
content-driven (``str.format`` over ``prompt_inputs`` for c22/c18/c23; a
literal ``{input}`` replace for c11), so a mutated or deserialized template
renders correctly. c19 does **not**: its ``ProbePair`` stores sentinel strings
(``"c19-naive"`` / ``"c19-ceiling"``) and its render dispatches by *Python
object identity* (``if template is NAIVE_TEMPLATE``). Any optimizer mutation
of the surfaced template, or even the unmutated naive template after a JSON
round-trip (value-equal but identity lost), makes the env render raise
``KeyError('unknown c19 probe template')`` -- so c19's cell is broken for both
optimization and serialize/deserialize.

This module fixes the *adapter's* fidelity without touching whetstone-envs (a
load-bearing checkout). :func:`probe_surface` returns a :class:`ProbeSurface`
whose ``naive_template`` / ``ceiling_template`` are genuine templates and whose
``render`` is content-driven:

* For c19 it binds real ``str.format`` templates whose slots are the public
  ``{grid}`` / ``{command}`` inputs plus a ``{fact_line}`` computed from the
  public ``fact_type`` input. The template head (the mutation target) is fully
  editable and the render depends only on template content -- never object
  identity -- so a mutated or round-tripped template renders. The rendered
  bytes are identical to the env's own ``render_naive`` / ``render_ceiling``
  (pinned by a per-fact-type equivalence test), so oracle fidelity is
  unchanged.
* For every other env it delegates verbatim to the env's own content-driven
  ``ProbePair`` (no behavior change).

Gold/oracle-only state can never be interpolated: c19's render formats against
only the public inputs (a ``{gold}`` template raises ``KeyError``), matching
the structural-leak proof the other envs already satisfy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone_envs.core import Instance, ProbePair


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
    render: Callable[[str, Instance], str]


def _from_probe_pair(probes: ProbePair) -> ProbeSurface:
    """Wrap an env's content-driven ``ProbePair`` verbatim.

    Used for c22/c11/c18/c23, whose ``ProbePair.render`` already keys off
    template content (``str.format`` over public inputs, or a literal
    ``{input}`` replace), so the surfaced template is already mutable and
    serialization-stable.
    """
    return ProbeSurface(
        naive_template=probes.naive_template,
        ceiling_template=probes.ceiling_template,
        render=probes.render,
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


def _c19_render(template: str, instance: Instance) -> str:
    """The content-driven c19 render for either probe template.

    Selects the per-fact-type ``fact_line`` table by template *content* (the
    ceiling marker), then formats ``template`` against the instance's public
    ``grid`` / ``command`` inputs plus that ``fact_line``. Restricted to
    public inputs: a ``{gold}`` (or any non-public) field raises ``KeyError``
    rather than interpolating oracle-only state. Dispatch is by content
    (``str.format`` + a substring marker), never object identity, so a mutated
    or round-tripped template renders.
    """
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
    fields = {
        "grid": inputs["grid"],
        "command": inputs["command"],
        "fact_line": fact_lines[fact_type],
    }
    return template.format(**fields)


def _c19_surface() -> ProbeSurface:
    """The real, mutable, serialization-stable c19 probe surface.

    The env's own head/fact-line text is imported so the rendered bytes are
    byte-for-byte identical to the env's ``render_naive`` / ``render_ceiling``
    (pinned by a test), preserving oracle fidelity while the surfaced template
    is a genuine format template rather than an identity sentinel.
    """
    from whetstone_envs.c19 import prompts as c19_prompts

    return ProbeSurface(
        naive_template=_c19_template(c19_prompts._NAIVE_HEAD),
        ceiling_template=_c19_template(c19_prompts._CEILING_HEAD),
        render=_c19_render,
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
    return _from_probe_pair(probes)


__all__ = [
    "ProbeSurface",
    "probe_surface",
]
