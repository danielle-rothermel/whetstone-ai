"""HumanEval code-compression environment family.

The ``code_comp`` package groups direct-generation (d1), encoder-decoder
(ed1), and behavioral-mutant (ed1m) modes that share HumanEval scoring,
dataset loading, and enc-dec rollout infrastructure. Legacy import paths
under ``whetstone.envs.d1``, ``ed1``, and ``ed1m`` remain as shims during
the migration toward a single ``code_comp`` env identity.
"""

from whetstone.envs.code_comp.constants import (
    ENCDEC_ENV_NAME,
)
from whetstone.envs.code_comp.dataset import (
    CodeCompTaskInstance,
    load_tasks,
)
from whetstone.envs.code_comp.modes.direct import (
    DirectExperiment,
    build_direct_experiment,
)
from whetstone.envs.code_comp.modes.encdec import (
    EncDecExperiment,
    EncDecTaskModelConfig,
    build_encdec_experiment,
)
from whetstone.envs.code_comp.modes.mutant import (
    MUTANT_ENV_NAME,
    MutantExperiment,
    build_mutant_experiment,
)
from whetstone.envs.code_comp.mutation_surface import validate_instruction_body
from whetstone.envs.code_comp.registry import (
    CodeCompMode,
    build_code_comp_experiment,
    code_comp_mode_for,
)
from whetstone.envs.code_comp.rollout.direct import DIRECT_ENV_NAME

__all__ = [
    "DIRECT_ENV_NAME",
    "ENCDEC_ENV_NAME",
    "MUTANT_ENV_NAME",
    "CodeCompMode",
    "CodeCompTaskInstance",
    "DirectExperiment",
    "EncDecExperiment",
    "EncDecTaskModelConfig",
    "MutantExperiment",
    "build_code_comp_experiment",
    "build_direct_experiment",
    "build_encdec_experiment",
    "build_mutant_experiment",
    "code_comp_mode_for",
    "load_tasks",
    "validate_instruction_body",
]
