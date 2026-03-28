"""Tests for scoring and report generation."""

import pytest

from accelbench.report import generate_report
from accelbench.types import RunRecord, TaskResult


def _make_result(
    task_id: str,
    passed: bool,
    tool_calls: int = 1,
    budget: int = 3,
    wall_time: float = 1.0,
    error: str | None = None,
    failure_reason: str | None = None,
    usage: dict | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        passed=passed,
        tool_calls=tool_calls,
        budget=budget,
        wall_time=wall_time,
        extracted_answer={"value": 42} if passed else None,
        error=error,
        failure_reason=failure_reason,
        usage=usage or {},
    )


class TestGenerateReport:
    def test_basic_structure(self):
        record = RunRecord(seed=42, config_path="test.yaml", adapter_name="Mock")
        record.results = [_make_result("1.1", passed=True)]
        report = generate_report(record)

        assert "summary" in report
        assert "per_tier" in report
        assert "tasks" in report
        assert "metadata" in report

    def test_pass_fail_counting(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=True),
            _make_result("1.2", passed=True),
            _make_result("1.3", passed=False, failure_reason="wrong_answer"),
        ]
        report = generate_report(record)
        s = report["summary"]
        assert s["total"] == 3
        assert s["passed"] == 2
        assert s["failed"] == 1
        assert s["errors"] == 0
        assert s["pass_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_error_counting(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=False, error="Setup failed", failure_reason="setup_error"),
            _make_result("1.2", passed=True),
        ]
        report = generate_report(record)
        s = report["summary"]
        assert s["errors"] == 1
        assert s["failed"] == 0
        assert s["passed"] == 1

    def test_per_tier_breakdown(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=True),
            _make_result("1.2", passed=False, failure_reason="wrong_answer"),
            _make_result("2.1", passed=True),
        ]
        report = generate_report(record)
        tiers = report["per_tier"]
        assert "1" in tiers
        assert tiers["1"]["total"] == 2
        assert tiers["1"]["passed"] == 1
        assert "2" in tiers
        assert tiers["2"]["total"] == 1
        assert tiers["2"]["passed"] == 1

    def test_failure_reasons(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=False, failure_reason="wrong_answer"),
            _make_result("1.2", passed=False, failure_reason="wrong_answer"),
            _make_result("1.3", passed=False, failure_reason="timeout"),
        ]
        report = generate_report(record)
        reasons = report["summary"]["failure_reasons"]
        assert reasons["wrong_answer"] == 2
        assert reasons["timeout"] == 1

    def test_token_usage_aggregation(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=True, usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
            _make_result("1.2", passed=True, usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}),
        ]
        report = generate_report(record)
        s = report["summary"]
        assert s["prompt_tokens"] == 300
        assert s["completion_tokens"] == 130
        assert s["total_tokens"] == 430

    def test_token_usage_missing(self):
        record = RunRecord(seed=42)
        record.results = [_make_result("1.1", passed=True)]
        report = generate_report(record)
        s = report["summary"]
        assert s["total_tokens"] == 0

    def test_task_details(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=True, tool_calls=1, budget=3, wall_time=2.5),
        ]
        report = generate_report(record)
        detail = report["tasks"][0]
        assert detail["task_id"] == "1.1"
        assert detail["passed"] is True
        assert detail["tool_calls"] == 1
        assert detail["budget"] == 3
        assert detail["wall_time"] == 2.5
        assert detail["efficiency"] == pytest.approx(1.0 - 1 / 3, abs=0.01)

    def test_task_detail_error_field(self):
        record = RunRecord(seed=42)
        record.results = [
            _make_result("1.1", passed=False, error="boom", failure_reason="crash"),
        ]
        report = generate_report(record)
        detail = report["tasks"][0]
        assert detail["error"] == "boom"
        assert detail["failure_reason"] == "crash"

    def test_metadata(self):
        record = RunRecord(
            seed=99,
            config_path="my_config.yaml",
            adapter_name="LiteLLM",
            model="gpt-4",
        )
        record.results = [_make_result("1.1", passed=True)]
        report = generate_report(record)
        m = report["metadata"]
        assert m["seed"] == 99
        assert m["config_path"] == "my_config.yaml"
        assert m["adapter"] == "LiteLLM"
        assert m["model"] == "gpt-4"

    def test_empty_results(self):
        record = RunRecord(seed=42)
        report = generate_report(record)
        s = report["summary"]
        assert s["total"] == 0
        assert s["pass_rate"] == 0.0


class TestEfficiency:
    def test_efficiency_zero_calls(self):
        r = TaskResult(
            task_id="1.1", passed=True, tool_calls=0, budget=3,
            wall_time=0.0, extracted_answer=None,
        )
        assert r.efficiency == 1.0

    def test_efficiency_full_budget(self):
        r = TaskResult(
            task_id="1.1", passed=True, tool_calls=3, budget=3,
            wall_time=0.0, extracted_answer=None,
        )
        assert r.efficiency == 0.0

    def test_efficiency_over_budget_clamps(self):
        r = TaskResult(
            task_id="1.1", passed=False, tool_calls=5, budget=3,
            wall_time=0.0, extracted_answer=None,
        )
        assert r.efficiency == 0.0

    def test_efficiency_partial(self):
        r = TaskResult(
            task_id="1.1", passed=True, tool_calls=1, budget=4,
            wall_time=0.0, extracted_answer=None,
        )
        assert r.efficiency == pytest.approx(0.75)
