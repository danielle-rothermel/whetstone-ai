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
    "AnchorArmPreview": "whetstone.eval.preview.anchor",
    "BaselinePreviewTranscript": "whetstone.eval.preview.anchor",
    "BaselineSweepTranscript": "whetstone.eval.preview.anchor",
    "run_baseline_preview": "whetstone.eval.preview.anchor",
    "run_baseline_sweep": "whetstone.eval.preview.anchor",
    "PreviewMetadata": "whetstone.eval.preview.preflight",
    "ScoringPreflight": "whetstone.eval.preview.preflight",
    "ScoredEvaluation": "whetstone.eval.preview.scored",
    "build_scored_evaluation": "whetstone.eval.preview.scored",
    "build_optim_eval_request": "whetstone.eval.preview.resolution",
    "build_measured_resolution": "whetstone.eval.preview.resolution",
    "evaluate_and_resolve": "whetstone.eval.preview.resolution",
    "load_aggregate_value": "whetstone.eval.preview.persisted",
    "load_component_traces": "whetstone.eval.preview.persisted",
    "load_evaluation_outputs": "whetstone.eval.preview.persisted",
}


def __getattr__(name: str):
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
