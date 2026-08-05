"""Package boundaries remain explicit and free of compatibility barrels."""

from typing import get_type_hints

import whetstone.optimization as optimization
from whetstone.coordination import official
from whetstone.evaluation import schema as evaluation_schema
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import TemplateRenderContract
from whetstone.experiment.reward import Reward
from whetstone.optimization import contracts
from whetstone.optimization.contracts import IntentResolution
from whetstone.optimization.harness import EvaluationService
from whetstone.optimization.miprov2 import study as miprov2_study
from whetstone.optimization.tools.admission import ToolCallStoreEntry


def test_optimization_package_is_thin() -> None:
    assert not hasattr(optimization, "__all__")
    assert not hasattr(optimization, "__getattr__")
    assert not hasattr(optimization, "OptimizationHarness")


def test_cross_boundary_contracts_have_canonical_owners() -> None:
    assert EvaluationBinding.__module__ == "whetstone.experiment.binding"
    assert (
        TemplateRenderContract.__module__ == "whetstone.experiment.candidate"
    )
    assert Reward.__module__ == "whetstone.experiment.reward"
    assert IntentResolution.__module__ == "whetstone.optimization.contracts"
    assert ToolCallStoreEntry.__module__ == (
        "whetstone.optimization.tools.admission"
    )


def test_former_cross_boundary_exports_are_absent() -> None:
    assert (
        not {
            "Candidate",
            "CandidateRef",
            "EvalConfigRef",
            "EvaluationBinding",
            "TemplateRenderContract",
            "candidate_reference",
            "EVALUATION_EVIDENCE_SCHEMA",
            "EVALUATION_FAILURE_SCHEMA",
        }
        & vars(contracts).keys()
    )
    assert (
        not {
            "EVALUATION_EVIDENCE_SCHEMA",
            "EVALUATION_FAILURE_SCHEMA",
            "REWARD_SCHEMA",
            "ROLLOUT_AGGREGATE_SCHEMA",
        }
        & vars(evaluation_schema).keys()
    )
    assert not hasattr(official, "TypedRef")
    assert (
        not {
            "EVALUATION_EVIDENCE_SCHEMA",
            "EVALUATION_FAILURE_SCHEMA",
            "REWARD_SCHEMA",
        }
        & vars(miprov2_study).keys()
    )


def test_graph_validation_protocol_stays_optimization_owned() -> None:
    hints = get_type_hints(EvaluationService.validate_resolution_graph)

    assert hints == {"resolution": IntentResolution, "return": type(None)}
