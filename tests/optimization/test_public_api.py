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


def test_boundary_modules_publish_exact_owned_exports() -> None:
    assert set(contracts.__all__) == {
        "INTENT_RESOLUTION_SCHEMA",
        "INTENT_RESOLUTION_SCHEMA_VERSION",
        "OPTIMIZATION_RESULT_SCHEMA",
        "OPTIMIZATION_RUN_SCHEMA",
        "OPTIMIZATION_RUN_SCHEMA_VERSION",
        "STEP_REQUEST_SCHEMA",
        "STEP_RESULT_SCHEMA",
        "BudgetDelta",
        "BudgetState",
        "EvaluationIntent",
        "IntentOutcome",
        "IntentResolution",
        "OptimizationProposal",
        "OptimizationResult",
        "OptimizationRun",
        "OptimizationRunRef",
        "OptimizationStepRequest",
        "OptimizationStepRequestRef",
        "OptimizationStepResult",
        "OptimizationStepResultRef",
        "OutputContract",
        "ResolutionClass",
        "ResolutionDetail",
        "StepKind",
        "StepMode",
        "StepStatus",
        "ToolEvidence",
        "optimization_result_reference",
        "optimization_run_reference",
        "step_request_reference",
        "step_result_reference",
    }
    assert set(evaluation_schema.__all__) == {
        "EVALUATION_COMPONENT_TRACES_SCHEMA",
        "EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION",
        "EVALUATION_EVIDENCE_SCHEMA_VERSION",
        "EVALUATION_OUTPUTS_SCHEMA",
        "EVALUATION_OUTPUTS_SCHEMA_VERSION",
        "CacheEvidence",
        "EvaluationComponentTraceRow",
        "EvaluationComponentTraces",
        "EvaluationComponentTracesRef",
        "EvaluationEvidence",
        "EvaluationEvidenceRef",
        "EvaluationFailureEvidence",
        "EvaluationFailureEvidenceRef",
        "EvaluationOutputRow",
        "EvaluationOutputsRecord",
        "RowAccounting",
    }
    assert set(official.__all__) == {
        "OFFICIAL_EVALUATION_RECORD_SCHEMA",
        "OFFICIAL_PLOT_MANIFEST_SCHEMA",
        "SELECTION_EVIDENCE_SCHEMA",
        "CompletenessDecision",
        "EvaluationAuthority",
        "MissingPlannedKeysError",
        "OfficialAggregationAccount",
        "OfficialEvaluationRecord",
        "OfficialFailurePolicy",
        "OfficialPlotManifest",
        "PlannedKeyResult",
        "RecordRevision",
        "RelabelingRefusedError",
        "SelectedRecordMapping",
        "SelectedRecordMappingEntry",
        "UnauthorizedOfficialWriteError",
        "account_planned_keys",
        "official_evaluation_record_reference",
        "official_plot_manifest_reference",
        "store_official_evaluation_record",
        "store_official_plot_manifest",
        "store_selection_evidence",
    }
    assert set(miprov2_study.__all__) == {
        "MIPROV2_ALGORITHM_VERSION",
        "MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA",
        "MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION",
        "MIPROV2_CANDIDATE_PROGRAM_SCHEMA",
        "MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION",
        "MIPROV2_CANDIDATE_RENDERING_SCHEMA",
        "MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION",
        "MIPROV2_REFERENCE_COMMIT",
        "MIPROV2_STUDY_SCHEMA",
        "MIPROV2_STUDY_SCHEMA_VERSION",
        "OPTUNA_VERSION",
        "BaselineObservation",
        "FullEvaluation",
        "Miprov2CandidateAssemblyBinding",
        "Miprov2CandidateRendering",
        "Miprov2ComponentSelection",
        "Miprov2EvaluationObservation",
        "Miprov2ParameterSpace",
        "Miprov2Study",
        "Miprov2StudySchedule",
        "Promotion",
        "PromotionCandidate",
        "SampleObservation",
        "StudySuggestion",
        "StudyTranscript",
        "StudyTranscriptMismatch",
        "TrialParams",
        "select_promotion",
    }


def test_graph_validation_protocol_stays_optimization_owned() -> None:
    hints = get_type_hints(EvaluationService.validate_resolution_graph)

    assert hints == {"resolution": IntentResolution, "return": type(None)}
