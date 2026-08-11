"""Legacy import path.

Implementation lives in whetstone.envs.code_comp.mutant.mutation.
"""

from whetstone.envs.code_comp.mutant.mutation import (
    ALL_FAMILIES,
    MutationError,
    MutationSite,
    OperatorFamily,
    apply_site,
    iter_sites,
)

__all__ = [
    "ALL_FAMILIES",
    "MutationError",
    "MutationSite",
    "OperatorFamily",
    "apply_site",
    "iter_sites",
]
