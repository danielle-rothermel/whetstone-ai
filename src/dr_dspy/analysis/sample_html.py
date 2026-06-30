"""HTML rendering for single-run enc-dec inspection reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer
from pygments.lexers.special import TextLexer

from dr_dspy.analysis.inspect import (
    PER_TEST_DISPLAY_LIMIT,
    PROMPTS_SOURCE,
    RunBundle,
    is_passing_run,
    node_attempt_by_id,
    summarize_test_results,
)
from dr_dspy.humaneval.task import EvaluationCaseStatus
from dr_dspy.lm.boundary import PromptMessage

ENCODER_NODE_ID = "encoder"
DECODER_NODE_ID = "decoder"
FAILING_STATUSES = frozenset(
    {
        EvaluationCaseStatus.FAILED,
        EvaluationCaseStatus.ERROR,
        EvaluationCaseStatus.TIMEOUT,
    }
)


def _highlight(text: str, *, language: str = "text") -> str:
    lexer = PythonLexer() if language == "python" else TextLexer()
    formatter = HtmlFormatter(nowrap=True, cssclass="highlight")
    return highlight(text or "", lexer, formatter)


def _escape(text: str) -> str:
    return html.escape(text or "")


def _format_messages(messages: tuple[PromptMessage, ...]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"[{message.role.value}]\n{message.content}")
    return "\n\n".join(parts)


def _card(title: str, body_html: str, *, subtitle: str = "") -> str:
    subtitle_html = (
        f'<div class="card-subtitle">{_escape(subtitle)}</div>'
        if subtitle
        else ""
    )
    return (
        f'<section class="card">'
        f'<h2>{_escape(title)}</h2>'
        f"{subtitle_html}"
        f'<div class="card-body">{body_html}</div>'
        f"</section>"
    )


def _code_block(text: str, *, language: str = "text") -> str:
    highlighted = _highlight(text, language=language)
    return f'<div class="code-block">{highlighted}</div>'


def _summary_card(bundle: RunBundle) -> str:
    dim_values = bundle.spec.dimensions.values
    passed = is_passing_run(bundle)
    outcome = "PASS" if passed else "FAIL / incomplete"
    score_status = (
        bundle.score_attempt.status.value
        if bundle.score_attempt is not None
        else "missing"
    )
    compression = (
        dim_values.get("compression_target")
        or dim_values.get("budget_ratio")
    )
    lines = [
        f"Experiment: {bundle.spec.experiment_name}",
        f"Sample index: {bundle.sample_index} / {bundle.sample_count - 1}",
        f"Task: {bundle.spec.task_id}",
        f"Outcome: {outcome}",
        f"Generation: {bundle.generation_run.status.value}",
        f"Score: {score_status}",
        f"Compression target: {compression}",
        f"Model: {bundle.spec.provider_axis.model}",
        f"Prediction ID: {bundle.spec.prediction_id}",
        f"Generation run ID: {bundle.generation_run.generation_run_id}",
    ]
    if bundle.score_attempt is not None:
        lines.append(
            f"Score attempt ID: {bundle.score_attempt.score_attempt_id}"
        )
    body = f"<pre class='plain'>{_escape(chr(10).join(lines))}</pre>"
    body += (
        f"<p class='note'>Prompts are <strong>{PROMPTS_SOURCE}</strong> "
        f"(not stored verbatim on node attempts).</p>"
    )
    return _card("Run summary", body)


def _ground_truth_card(bundle: RunBundle) -> str:
    inputs = bundle.spec.task.inputs.values
    gt_code = str(inputs.get("gt_code") or "")
    prompt = inputs.get("prompt")
    parts = [_code_block(gt_code, language="python")]
    if prompt:
        parts.append(
            f"<h3>Task prompt</h3>{_code_block(str(prompt), language='text')}"
        )
    budget = inputs.get("budget")
    if budget is not None:
        parts.append(f"<p>Budget: {_escape(str(budget))} characters</p>")
    return _card("Ground truth", "".join(parts))


def _prompt_card(
    title: str,
    messages: tuple[PromptMessage, ...] | None,
    *,
    error: str | None = None,
) -> str:
    if messages is None:
        body = f"<p class='error'>{_escape(error or 'Prompt unavailable')}</p>"
        return _card(title, body, subtitle=PROMPTS_SOURCE)
    return _card(
        title,
        _code_block(_format_messages(messages), language="text"),
        subtitle=PROMPTS_SOURCE,
    )


def _node_output_card(
    title: str,
    bundle: RunBundle,
    node_id: str,
    output_field: str,
) -> str:
    attempt = node_attempt_by_id(bundle, node_id)
    if attempt is None:
        return _card(title, "<p class='error'>Node attempt missing</p>")
    if attempt.output is None:
        failure = attempt.failure.message if attempt.failure else "No output"
        return _card(
            title,
            (
                f"<p class='error'>Status: {attempt.status.value}. "
                f"{_escape(failure)}</p>"
            ),
        )
    value = attempt.output.values.get(output_field)
    if value is None:
        return _card(
            title,
            f"<p class='error'>Output field {output_field!r} missing</p>",
        )
    language = "python" if output_field == "code" else "text"
    return _card(title, _code_block(str(value), language=language))


def _extraction_card(bundle: RunBundle) -> str:
    score_attempt = bundle.score_attempt
    if score_attempt is None or score_attempt.extracted_code is None:
        return _card("Extraction", "<p class='error'>No score attempt</p>")
    extracted = score_attempt.extracted_code
    meta = extracted.metadata or {}
    lines = [
        f"Method: {extracted.extraction_method}",
        f"Parser: {extracted.parser_profile_id}@{extracted.parser_version}",
        f"Compile OK: {meta.get('compile_ok')}",
    ]
    if meta.get("compile_error"):
        lines.append(f"Compile error: {meta['compile_error']}")
    if meta.get("extraction_error"):
        lines.append(f"Extraction error: {meta['extraction_error']}")
    header = f"<pre class='plain'>{_escape(chr(10).join(lines))}</pre>"
    raw = extracted.raw_generation or ""
    raw_block = _code_block(raw, language="python")
    body = header + "<h3>Raw generation</h3>" + raw_block
    return _card("Extraction", body)


def _scored_code_card(bundle: RunBundle) -> str:
    score_attempt = bundle.score_attempt
    if score_attempt is None or score_attempt.extracted_code is None:
        return _card("Scored code", "<p class='error'>No extracted code</p>")
    code = score_attempt.extracted_code.extracted_code or ""
    outcome = (
        score_attempt.generated_code_outcome.value
        if score_attempt.generated_code_outcome is not None
        else "unknown"
    )
    score = score_attempt.score
    subtitle = f"outcome={outcome}, score={score}"
    return _card(
        "Scored code",
        _code_block(code, language="python"),
        subtitle=subtitle,
    )


def _tests_card(bundle: RunBundle) -> str:
    score_attempt = bundle.score_attempt
    if score_attempt is None:
        return _card("Tests", "<p class='error'>No score attempt</p>")
    summary = summarize_test_results(score_attempt)
    header_lines = [
        f"Total: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Error/timeout: {summary['error']}",
    ]
    if summary["truncated"]:
        header_lines.append(
            "Note: per_test_results may be truncated in metrics"
        )
    body_parts = [
        f"<pre class='plain'>{_escape(chr(10).join(header_lines))}</pre>"
    ]
    failing = [
        result
        for result in score_attempt.per_test_results
        if result.status in FAILING_STATUSES
    ]
    if failing:
        body_parts.append("<h3>Failing tests</h3><ul class='fail-list'>")
        for result in failing[:PER_TEST_DISPLAY_LIMIT]:
            body_parts.append(
                "<li>"
                f"<strong>{_escape(result.test_id)}</strong> "
                f"({_escape(result.status.value)}): "
                f"{_escape(result.message)}"
                "</li>"
            )
        body_parts.append("</ul>")
        if len(failing) > PER_TEST_DISPLAY_LIMIT:
            remaining = len(failing) - PER_TEST_DISPLAY_LIMIT
            body_parts.append(
                f"<p>… and {remaining} more failures (see JSON bundle)</p>"
            )
    return _card("Tests", "".join(body_parts))


def _pygments_css() -> str:
    return HtmlFormatter().get_style_defs(".highlight")


def render_sample_html(
    bundle: RunBundle,
    metadata: dict[str, Any],
    reconstructed_prompts: dict[str, tuple[PromptMessage, ...]],
    reconstruction_errors: list[str],
    *,
    json_path: Path,
) -> str:
    encoder_prompt = reconstructed_prompts.get(ENCODER_NODE_ID)
    decoder_prompt = reconstructed_prompts.get(DECODER_NODE_ID)
    encoder_error = next(
        (err for err in reconstruction_errors if err.startswith("encoder")),
        None,
    )
    decoder_error = next(
        (err for err in reconstruction_errors if err.startswith("decoder")),
        None,
    )

    cards = [
        _summary_card(bundle),
        _ground_truth_card(bundle),
        _prompt_card("Encoder prompt", encoder_prompt, error=encoder_error),
        _node_output_card(
            "Encoder output",
            bundle,
            ENCODER_NODE_ID,
            "description",
        ),
        _prompt_card("Decoder prompt", decoder_prompt, error=decoder_error),
        _node_output_card(
            "Decoder output",
            bundle,
            DECODER_NODE_ID,
            "code",
        ),
        _extraction_card(bundle),
        _scored_code_card(bundle),
        _tests_card(bundle),
    ]

    page_title = (
        f"{_escape(bundle.spec.experiment_name)} "
        f"sample {bundle.sample_index}"
    )
    jq_example = (
        f"jq '.spec.prediction_id' {_escape(str(json_path))}"
    )
    metadata_json = json.dumps(metadata, indent=2, default=str)
    metadata_script = json.dumps(metadata, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{page_title}</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      margin: 0;
      padding: 1rem;
      background: #f5f5f5;
      color: #111;
    }}
    h1 {{ margin-top: 0; }}
    .pipeline {{
      display: flex;
      flex-direction: row;
      gap: 1rem;
      overflow-x: auto;
      padding-bottom: 1rem;
    }}
    .card {{
      min-width: 28rem;
      max-width: 36rem;
      flex: 0 0 auto;
      background: #fff;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 0.75rem 1rem;
    }}
    .card h2 {{
      margin: 0 0 0.5rem;
      font-size: 1rem;
    }}
    .card-subtitle {{
      font-size: 0.8rem;
      color: #555;
      margin-bottom: 0.5rem;
    }}
    .card-body {{ font-size: 0.85rem; }}
    .code-block {{
      overflow-x: auto;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 0.25rem;
      background: #fafafa;
    }}
    pre.plain {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fafafa;
      border: 1px solid #ddd;
      padding: 0.5rem;
      border-radius: 4px;
    }}
    .note {{ font-size: 0.85rem; color: #444; }}
    .error {{ color: #a00; }}
    .metadata {{
      margin-top: 1.5rem;
      background: #fff;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 1rem;
    }}
    .metadata pre {{
      overflow-x: auto;
      white-space: pre;
      font-size: 0.75rem;
    }}
    ul.fail-list {{
      padding-left: 1.2rem;
      font-size: 0.8rem;
    }}
    {_pygments_css()}
  </style>
</head>
<body>
  <h1>{_escape(bundle.spec.task_id)} — sample {bundle.sample_index}</h1>
  <p>Experiment: {_escape(bundle.spec.experiment_name)}</p>
  <div class="pipeline">
    {"".join(cards)}
  </div>
  <section class="metadata">
    <h2>Debug metadata (copy for jq)</h2>
    <p>Sibling JSON file: <code>{_escape(str(json_path))}</code></p>
    <p>Example: <code>{jq_example}</code></p>
    <pre id="debug-metadata">{_escape(metadata_json)}</pre>
  </section>
  <script type="application/json" id="debug-metadata-json">
{metadata_script}
  </script>
</body>
</html>
"""


def write_sample_report(
    *,
    bundle: RunBundle,
    metadata: dict[str, Any],
    reconstructed_prompts: dict[str, tuple[PromptMessage, ...]],
    reconstruction_errors: list[str],
    html_path: Path,
    json_path: Path,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_sample_html(
        bundle,
        metadata,
        reconstructed_prompts,
        reconstruction_errors,
        json_path=json_path,
    )
    html_path.write_text(html_content, encoding="utf-8")
    json_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
