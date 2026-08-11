"""HumanEval code-compression environment family.

The ``code_comp`` package unifies direct-generation, encoder-decoder, and
behavioral-mutant HumanEval compression experiments under one env identity
(``code_comp``) with mode-specific subsets (``direct``, ``encdec``,
``encdec_mutant``).
"""

from whetstone.envs.code_comp.constants import CODE_COMP_ENV_NAME
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
    MutantExperiment,
    build_mutant_experiment,
)
from whetstone.envs.code_comp.mutation_surface import validate_instruction_body
from whetstone.envs.code_comp.registry import (
    CodeCompMode,
    build_code_comp_experiment,
    code_comp_identity_prefix,
    code_comp_mode_for,
)

__all__ = [
    "CODE_COMP_ENV_NAME",
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
    "code_comp_identity_prefix",
    "code_comp_mode_for",
    "load_tasks",
    "validate_instruction_body",
]
