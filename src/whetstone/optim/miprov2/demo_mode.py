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
    Whetstone extension: it bootstraps fewshot-sized demonstration pools and
    grounds instruction proposals in them, but excludes the demo dimension
    from the parameter space and never attaches a demo set to a candidate.
    Both non-searching modes share the zeroshot auto-mode trial/instruct
    arm.
    """

    #: DSPy's default: bootstrap demos, search instruction x demo-set, and
    #: render the selected demos into the candidate.
    FEWSHOT = auto()
    #: DSPy's 0-shot mode: control maxima stay 0/0 and demos stay out of
    #: the study, but the run still bootstraps 3/0 demos to ground
    #: instruction proposals and then discards them.
    ZEROSHOT = auto()
    #: Whetstone extension: bootstrap fewshot-sized pools to ground
    #: proposals, search instructions only, and never render demos into a
    #: candidate.
    GROUND_ONLY = auto()

    @property
    def bootstraps(self) -> bool:
        """Whether this mode runs bootstrap evaluations through the engine."""
        return True

    @property
    def searches_demos(self) -> bool:
        """Whether the demo dimension enters the study's parameter space."""
        return self is Miprov2DemoMode.FEWSHOT

    @property
    def is_faithful_dspy(self) -> bool:
        """Whether this mode reproduces frozen DSPy MIPROv2 behavior."""
        return self is not Miprov2DemoMode.GROUND_ONLY
