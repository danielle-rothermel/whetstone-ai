"""Evaluation Authority, certification, and publication (Workstream 9).

The Whetstone official write path: a named :class:`EvaluationAuthority`
principal that is the only issuer of official Evaluation Contexts, immutable
:class:`OfficialEvaluationRecord` certifications over ordinary Rollout Result
references, and immutable :class:`OfficialPlotManifest` publications.

Load-bearing guarantees:

* **No relabeling.** Internal evaluation evidence can never be relabeled or
  copied to official because its config Identity Hashes match an official run;
  :meth:`EvaluationAuthority.certify` refuses internal-role Contexts
  (:class:`RelabelingRefusedError`).
* **Every planned key accounted for.** :func:`account_planned_keys` produces a
  complete account where every planned Rollout Execution Key is present or an
  explicit, visible missing row — never dropped.
* **Mandatory ordered mapping.** Both official records preserve the ordered
  selected-record -> ``graph_hash`` -> shared planned/result-key set ->
  aggregate mapping (:class:`SelectedRecordMapping`), so two selected records
  that share one graph stay separately attributable.
* **Record-local provenance.** Each record owns typed provenance fields; there
  is no universal Provenance class.
"""

from whetstone.authority.aggregation import (
    MissingPlannedKeysError,
    OfficialAggregationAccount,
    OfficialFailurePolicy,
    account_planned_keys,
)
from whetstone.authority.authority import (
    EvaluationAuthority,
    RelabelingRefusedError,
    UnauthorizedOfficialWriteError,
)
from whetstone.authority.mapping import (
    SelectedRecordMapping,
    SelectedRecordMappingEntry,
)
from whetstone.authority.records import (
    OFFICIAL_EVALUATION_RECORD_SCHEMA,
    OFFICIAL_PLOT_MANIFEST_SCHEMA,
    CompletenessDecision,
    OfficialEvaluationRecord,
    OfficialPlotManifest,
    PlannedKeyResult,
    RecordRevision,
)
from whetstone.authority.reference import TypedContentRef
from whetstone.authority.store import (
    SELECTION_EVIDENCE_SCHEMA,
    official_evaluation_record_reference,
    official_plot_manifest_reference,
    store_official_evaluation_record,
    store_official_plot_manifest,
    store_selection_evidence,
)

__all__ = [
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
    "TypedContentRef",
    "UnauthorizedOfficialWriteError",
    "account_planned_keys",
    "official_evaluation_record_reference",
    "official_plot_manifest_reference",
    "store_official_evaluation_record",
    "store_official_plot_manifest",
    "store_selection_evidence",
]
