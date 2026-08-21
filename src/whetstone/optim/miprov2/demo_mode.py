"""How demonstrations participate in one MIPROv2 run.

This is a leaf module: the demo mode is read by the control, the bootstrap
and proposal state machines, and the study's parameter space, all of which
sit at different depths of the MIPROv2 import graph.
"""

from __future__ import annotations

from enum import UNIQUE, StrEnum, auto, verify

__all__ = ["Miprov2DemoMode"]


@verify(UNIQUE)
class Miprov2DemoMode(StrEnum):
    """The three demonstration regimes a MIPROv2 run may take.

    ``FEWSHOT`` and ``ZEROSHOT`` are the two faithful DSPy behaviors and keep
    ``algorithm_version`` at ``dspy_miprov2/v2``. ``GROUND_ONLY`` is a
    Whetstone extension: it bootstraps demonstrations and grounds instruction
    proposals in them exactly as ``FEWSHOT`` does, but excludes the demo
    dimension from the parameter space and never attaches a demo set to a
    candidate, so the study optimizes instructions alone over demo-grounded
    proposals.
    """

    #: DSPy's default: bootstrap demos, search instruction x demo-set, and
    #: render the selected demos into the candidate.
    FEWSHOT = auto()
    #: DSPy's 0-shot mode: no bootstrapping, and both demo maxima are zero.
    ZEROSHOT = auto()
    #: Whetstone extension: bootstrap to ground proposals, search
    #: instructions only, and never render demos into a candidate.
    GROUND_ONLY = auto()

    @property
    def bootstraps(self) -> bool:
        """Whether this mode runs bootstrap evaluations through the engine."""
        return self is not Miprov2DemoMode.ZEROSHOT

    @property
    def searches_demos(self) -> bool:
        """Whether the demo dimension enters the study's parameter space."""
        return self is Miprov2DemoMode.FEWSHOT

    @property
    def is_faithful_dspy(self) -> bool:
        """Whether this mode reproduces frozen DSPy MIPROv2 behavior."""
        return self is not Miprov2DemoMode.GROUND_ONLY
