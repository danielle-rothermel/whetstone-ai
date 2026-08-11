"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.mutant.oracle.
"""

from whetstone.envs.code_comp.mutant.oracle import (
    MutantScore,
    score_ed1m_reconstruction,
)

__all__ = ["MutantScore", "score_ed1m_reconstruction"]
