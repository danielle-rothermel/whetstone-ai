from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dr_dspy.eval_failures import (
    FailureClass,
    PermanentFailureError,
    RateLimitedFailureError,
    ResourceExhaustionFailureError,
    TransientFailureError,
    policy,
    should_retry_step,
    summarize_exception,
)


def _fake_module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


@pytest.mark.parametrize(
    ("error", "expected_class", "expected_retry"),
    [
        (
            PermanentFailureError("permanent"),
            FailureClass.PERMANENT,
            False,
        ),
        (
            TransientFailureError("transient"),
            FailureClass.TRANSIENT,
            True,
        ),
        (
            RateLimitedFailureError("rate limited"),
            FailureClass.RATE_LIMITED,
            True,
        ),
        (
            ResourceExhaustionFailureError("resource exhausted"),
            FailureClass.RESOURCE_EXHAUSTION,
            False,
        ),
    ],
)
def test_explicit_failure_classes_define_recovery_and_retry_policy(
    error: BaseException,
    expected_class: FailureClass,
    expected_retry: bool,
) -> None:
    summary = summarize_exception(error)

    assert summary.failure_class is expected_class
    assert summary.is_recoverable is (
        expected_class
        in {
            FailureClass.TRANSIENT,
            FailureClass.RATE_LIMITED,
            FailureClass.RESOURCE_EXHAUSTION,
        }
    )
    assert should_retry_step(error) is expected_retry


def test_explicit_failure_summary_preserves_metadata() -> None:
    error = PermanentFailureError("bad input", metadata={"task_id": "x"})

    summary = summarize_exception(error)

    assert summary.failure_class is FailureClass.PERMANENT
    assert summary.failure_metadata == {"task_id": "x"}


def test_lazy_openai_rate_limit_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRateLimitError(Exception):
        pass

    module_name = "tests.fake_openai_rate_limit"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, RateLimitError=FakeRateLimitError),
    )
    monkeypatch.setattr(policy, "OPENAI_MODULE", module_name)

    error = FakeRateLimitError("slow down")

    assert policy.classify_exception(error) is FailureClass.RATE_LIMITED
    assert policy.should_retry_step(error) is True


@pytest.mark.parametrize(
    ("status_code", "expected_class", "expected_retry"),
    [
        (429, FailureClass.RATE_LIMITED, True),
        (500, FailureClass.TRANSIENT, True),
        (409, FailureClass.TRANSIENT, True),
        (400, FailureClass.PERMANENT, False),
    ],
)
def test_lazy_openai_status_classification(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_class: FailureClass,
    expected_retry: bool,
) -> None:
    class FakeAPIStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"status {status_code}")
            self.status_code = status_code

    module_name = "tests.fake_openai_status"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, APIStatusError=FakeAPIStatusError),
    )
    monkeypatch.setattr(policy, "OPENAI_MODULE", module_name)

    error = FakeAPIStatusError(status_code)

    assert policy.classify_exception(error) is expected_class
    assert policy.should_retry_step(error) is expected_retry


@pytest.mark.parametrize(
    ("status_code", "expected_class", "expected_retry"),
    [
        (429, FailureClass.RATE_LIMITED, True),
        (503, FailureClass.TRANSIENT, True),
        (425, FailureClass.TRANSIENT, True),
        (404, FailureClass.PERMANENT, False),
    ],
)
def test_lazy_httpx_status_classification(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_class: FailureClass,
    expected_retry: bool,
) -> None:
    class FakeHTTPStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"status {status_code}")
            self.response = SimpleNamespace(status_code=status_code)

    module_name = "tests.fake_httpx"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, HTTPStatusError=FakeHTTPStatusError),
    )
    monkeypatch.setattr(policy, "HTTPX_MODULE", module_name)

    error = FakeHTTPStatusError(status_code)

    assert policy.classify_exception(error) is expected_class
    assert policy.should_retry_step(error) is expected_retry


def test_lazy_psycopg_operational_error_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOperationalError(Exception):
        pass

    module_name = "tests.fake_psycopg"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, OperationalError=FakeOperationalError),
    )
    monkeypatch.setattr(policy, "PSYCOPG_MODULE", module_name)

    error = FakeOperationalError("connection lost")

    assert policy.classify_exception(error) is FailureClass.TRANSIENT
    assert policy.should_retry_step(error) is True


def test_lazy_dbos_retry_wrapper_unwraps_last_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDBOSMaxStepRetriesExceeded(Exception):
        def __init__(self, errors: list[BaseException]) -> None:
            super().__init__("max retries")
            self.errors = errors

    module_name = "tests.fake_dbos_error"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(
            module_name,
            DBOSMaxStepRetriesExceeded=FakeDBOSMaxStepRetriesExceeded,
        ),
    )
    monkeypatch.setattr(policy, "DBOS_ERROR_MODULE", module_name)

    wrapper = FakeDBOSMaxStepRetriesExceeded(
        [PermanentFailureError("first"), TransientFailureError("last")]
    )

    assert policy.unwrap_exception(wrapper).args == ("last",)
    assert policy.classify_exception(wrapper) is FailureClass.TRANSIENT
    assert policy.should_retry_step(wrapper) is True


def test_policy_import_does_not_load_runtime_exception_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import dr_dspy.eval_failures.policy; "
                "print(any(name in sys.modules for name in "
                "('dbos._error', 'openai', 'httpx', 'psycopg')))"
            ),
        ],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )

    assert completed.stdout.strip() == "False"


def test_recording_import_defers_psycopg_jsonb() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import dr_dspy.eval_failures.recording; "
                "print('psycopg.types.json' in sys.modules)"
            ),
        ],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )

    assert completed.stdout.strip() == "False"
