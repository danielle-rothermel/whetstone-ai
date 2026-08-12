__all__ = [
    "AnchorConfigPreview",
    "BaselinePreviewTranscript",
    "BaselineSweepTranscript",
    "PreviewMetadata",
    "ScoredCandidate",
    "ScoringPreflight",
    "build_evaluation_intent",
    "build_measured_resolution",
    "build_scored_candidate",
    "evaluate_and_resolve",
    "load_aggregate_value",
    "load_component_traces",
    "load_evaluation_outputs",
    "preview_evaluation_binding",
    "run_baseline_preview",
    "run_baseline_sweep",
]

_MODULE_EXPORTS = {
    "AnchorConfigPreview": "whetstone.evaluation.preview.anchor",
    "BaselinePreviewTranscript": "whetstone.evaluation.preview.anchor",
    "BaselineSweepTranscript": "whetstone.evaluation.preview.anchor",
    "run_baseline_preview": "whetstone.evaluation.preview.anchor",
    "run_baseline_sweep": "whetstone.evaluation.preview.anchor",
    "PreviewMetadata": "whetstone.evaluation.preview.preflight",
    "ScoringPreflight": "whetstone.evaluation.preview.preflight",
    "ScoredCandidate": "whetstone.evaluation.preview.scored",
    "build_scored_candidate": "whetstone.evaluation.preview.scored",
    "build_evaluation_intent": "whetstone.evaluation.preview.resolution",
    "build_measured_resolution": "whetstone.evaluation.preview.resolution",
    "evaluate_and_resolve": "whetstone.evaluation.preview.resolution",
    "load_aggregate_value": "whetstone.evaluation.preview.persisted",
    "load_component_traces": "whetstone.evaluation.preview.persisted",
    "load_evaluation_outputs": "whetstone.evaluation.preview.persisted",
    "preview_evaluation_binding": "whetstone.evaluation.preview.binding",
}


def __getattr__(name: str):
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
