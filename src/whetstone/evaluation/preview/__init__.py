__all__ = [
    "AnchorArmPreview",
    "BaselinePreviewTranscript",
    "BaselineSweepTranscript",
    "PreviewMetadata",
    "ScoredEvaluation",
    "ScoringPreflight",
    "build_measured_resolution",
    "build_optim_eval_request",
    "build_scored_evaluation",
    "evaluate_and_resolve",
    "load_aggregate_value",
    "load_component_traces",
    "load_evaluation_outputs",
    "run_baseline_preview",
    "run_baseline_sweep",
]

_MODULE_EXPORTS = {
    "AnchorArmPreview": "whetstone.evaluation.preview.anchor",
    "BaselinePreviewTranscript": "whetstone.evaluation.preview.anchor",
    "BaselineSweepTranscript": "whetstone.evaluation.preview.anchor",
    "run_baseline_preview": "whetstone.evaluation.preview.anchor",
    "run_baseline_sweep": "whetstone.evaluation.preview.anchor",
    "PreviewMetadata": "whetstone.evaluation.preview.preflight",
    "ScoringPreflight": "whetstone.evaluation.preview.preflight",
    "ScoredEvaluation": "whetstone.evaluation.preview.scored",
    "build_scored_evaluation": "whetstone.evaluation.preview.scored",
    "build_optim_eval_request": "whetstone.evaluation.preview.resolution",
    "build_measured_resolution": "whetstone.evaluation.preview.resolution",
    "evaluate_and_resolve": "whetstone.evaluation.preview.resolution",
    "load_aggregate_value": "whetstone.evaluation.preview.persisted",
    "load_component_traces": "whetstone.evaluation.preview.persisted",
    "load_evaluation_outputs": "whetstone.evaluation.preview.persisted",
}


def __getattr__(name: str):
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
