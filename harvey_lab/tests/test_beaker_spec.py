"""Tests for the Beaker prompt-optimization spec (``harvey_lab/beaker_spec.py``).

These cover the two things that silently break a prompt-optimization
integration: a dataset contract that drifts from the exported rows, and
candidate prompts that never actually reach the agent. The rollout test runs the
**real** ``HarveyLabAgent`` against a fixture task with a scripted Stirrup client
and a stub rubric judge, so it asserts the candidate prompts on the wire without
a single network call.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from beaker import Case, CaseResult, DatasetRowContext, OptimizationContext, validate_spec
from beaker.sdk import RolloutContext

from harvey_lab.agent.agent import HarveyLabAgent
from harvey_lab.agent.prompts import load_harvey_lab_prompts
from harvey_lab.agent.workspace import task_source_from_dir
from harvey_lab.config import HarveyLabConfig
from harvey_lab.evaluation.scoring import ALL_PASS_FIELD, CRITERION_PASS_RATE_FIELD
from tests.test_units import _fee_judge, _local_exec_factory, _ScriptedClient, _write_task


RECIPE_ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_ROOT = RECIPE_ROOT.parent
SPEC_PATH = RECIPE_ROOT / "beaker_spec.py"
DATASET_DIR = COOKBOOK_ROOT / ".beaker" / "dataset"

SYSTEM_MARKER = "CANDIDATE-SYSTEM-PROMPT-MARKER"
TEMPLATE_MARKER = "CANDIDATE-TASK-TEMPLATE-MARKER"


@pytest.fixture(scope="module")
def spec_module() -> ModuleType:
    """Import the spec by path (``.beaker`` is not an importable package name)."""
    module_spec = importlib.util.spec_from_file_location("beaker_spec_under_test", SPEC_PATH)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    # dataclasses resolve their annotations through sys.modules, so register the
    # module before executing it.
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def offline_preflight(spec_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record the loader's pre-flight fetches instead of hitting GitHub."""
    calls: list[tuple[str, str]] = []

    def _fake(task_ids: Any, *, commit: str, cache_dir: Path | None = None) -> Path:
        del cache_dir
        calls.extend((str(task_id), commit) for task_id in task_ids)
        return Path("/nonexistent-tasks-root")

    monkeypatch.setattr(spec_module, "ensure_task_dirs", _fake)
    return calls


@pytest.fixture
def lab_task_root(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    _write_task(
        root,
        "contracts/t1",
        criteria=[
            {
                "id": "C1",
                "title": "States the fee",
                "match_criteria": "Mentions $50,000.",
                "deliverables": ["memo.md"],
            },
            {"id": "C2", "title": "Cites source", "match_criteria": "Cites notes.txt.", "deliverables": ["memo.md"]},
        ],
        deliverables={"memo.md": "Memo"},
    )
    return root


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "contracts/t1",
        "input": {
            "task_id": "contracts/t1",
            "title": "task contracts/t1",
            "work_type": "analyze",
            "instructions": "Summarize the termination fee and cite the source document.",
            "deliverables": {"memo.md": "Memo"},
            "documents": ["notes.txt"],
        },
        "expected": {
            "criteria": [
                {"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000.", "deliverables": []},
            ]
        },
        "metadata": {"practice_area": "contracts"},
    }
    row.update(overrides)
    return row


# ─── seed targets ─────────────────────────────────────────────────────


def test_seed_targets_are_the_prompts_the_recipe_ships(spec_module: ModuleType) -> None:
    system_prompt, task_template = load_harvey_lab_prompts()
    assert spec_module._seed_targets().to_dict() == {
        "system_prompt": system_prompt,
        "task_template": task_template,
    }


# ─── dataset contract ─────────────────────────────────────────────────


def test_loader_builds_cases_from_a_row(spec_module: ModuleType) -> None:
    loader = spec_module.LabTaskDataLoader()
    context = DatasetRowContext(split="train", line_number=1)
    cases = list(loader.iter_cases(loader.parse_row(_row(), context), context))
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "contracts/t1"
    assert case.group_key == "contracts"
    assert case.input["instructions"].startswith("Summarize")
    assert [criterion["id"] for criterion in case.ground_truth["criteria"]] == ["C1"]


def test_loader_materializes_the_task_before_any_rollout(
    spec_module: ModuleType, offline_preflight: list[tuple[str, str]]
) -> None:
    """Building the cases is what downloads the tasks, not running them.

    Dataset loading finishes before the engine schedules a single rollout, so a
    fetch here is the run's one-time pre-flight; a fetch inside ``run_case``
    would instead be dozens of rollouts racing on the same task tree.
    """
    loader = spec_module.LabTaskDataLoader()
    context = DatasetRowContext(split="train", line_number=1)
    row = _row(metadata={"practice_area": "contracts", "harvey_labs_commit": "deadbeef"})
    list(loader.iter_cases(loader.parse_row(row, context), context))
    assert offline_preflight == [("contracts/t1", "deadbeef")]


def test_loader_pre_flight_failure_does_not_abort_dataset_loading(
    spec_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A task that cannot be pre-fetched must fail its own case in run_case, not
    # take the whole run down while the dataset is still being read.
    def _boom(*_args: Any, **_kwargs: Any) -> Path:
        raise RuntimeError("github is down")

    monkeypatch.setattr(spec_module, "ensure_task_dirs", _boom)
    loader = spec_module.LabTaskDataLoader()
    context = DatasetRowContext(split="train", line_number=1)
    cases = list(loader.iter_cases(loader.parse_row(_row(), context), context))
    assert [case.case_id for case in cases] == ["contracts/t1"]


@pytest.mark.parametrize(
    "row",
    [
        _row(expected={"criteria": []}),
        _row(expected={"criteria": [{"id": "C1", "title": "no standard"}]}),
        _row(input={"task_id": "contracts/t1", "instructions": "  "}),
        _row(id="", input={"instructions": "Summarize the fee."}),
        _row(expected=["not", "an", "object"]),
    ],
)
def test_loader_rejects_unusable_rows(spec_module: ModuleType, row: dict[str, Any]) -> None:
    loader = spec_module.LabTaskDataLoader()
    with pytest.raises((ValueError, TypeError)):
        loader.parse_row(row, DatasetRowContext(split="train", line_number=1))


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="local dataset export not present")
def test_exported_dataset_matches_the_declared_contract(spec_module: ModuleType) -> None:
    loader = spec_module.LabTaskDataLoader()
    for split in ("train", "val"):
        path = DATASET_DIR / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows, f"{path} is empty"
        for index, raw in enumerate(rows, start=1):
            context = DatasetRowContext(split=split, path=str(path), line_number=index)
            cases = list(loader.iter_cases(loader.parse_row(raw, context), context))
            assert len(cases) == 1
            assert cases[0].ground_truth["criteria"], f"{context.label()}: unscoreable row"


# ─── rollout ──────────────────────────────────────────────────────────


def _patch_for_local_rollout(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
    *,
    body: str,
    submit_nothing: bool = False,
) -> list[list[Any]]:
    """Wire ``_run_case`` to a fixture task, a scripted client and a stub judge.

    Everything under test stays real: the actual ``HarveyLabAgent``, its prompt
    rendering, its ``code_exec`` workspace and the recipe's rubric aggregation.
    Returns the list the client records each turn's messages into.
    """
    seen_messages: list[list[Any]] = []

    class _RecordingClient(_ScriptedClient):
        async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
            seen_messages.append(list(messages))
            return await super().generate(messages, tools)

    class _EmptySubmissionClient(_RecordingClient):
        """Runs to completion and finishes without submitting any file."""

        async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
            from stirrup.core.models import AssistantMessage, ToolCall

            seen_messages.append(list(messages))
            call = ToolCall(
                name="finish",
                arguments=json.dumps({"summary": "Nothing to submit.", "paths": []}),
                tool_call_id="tc-1",
            )
            return AssistantMessage(content="", tool_calls=[call])

    def _model_factory(*_args: Any) -> Any:
        client = _EmptySubmissionClient if submit_nothing else _RecordingClient
        return client(deliverable_body=body, deliverable_name="memo.md")

    record = spec_module.load_records(lab_task_root, task_ids=["contracts/t1"])[0]
    monkeypatch.setattr(spec_module, "_record_for_case", lambda case, **_kw: (record, lab_task_root))
    monkeypatch.setattr(
        spec_module,
        "_config_for_runtime",
        lambda _runtime: (
            HarveyLabConfig(max_turns=5, enable_view_image=False),
            spec_module._TaskModelRouting(),
        ),
    )
    monkeypatch.setattr(
        spec_module,
        "HarveyLabAgent",
        partial(HarveyLabAgent, model_factory=_model_factory, exec_provider_factory=_local_exec_factory),
    )
    monkeypatch.setattr(spec_module, "build_rubric_judge", lambda *_a, **_kw: _fee_judge)
    return seen_messages


def _candidate_prompts(spec_module: ModuleType) -> Any:
    system_prompt, task_template = load_harvey_lab_prompts()
    return spec_module.optimization_targets_from_prompts(
        {
            "system_prompt": f"{system_prompt}\n\n{SYSTEM_MARKER}",
            "task_template": f"{task_template}\n\n{TEMPLATE_MARKER}",
        }
    )


def test_run_case_sends_the_candidate_prompts_to_the_agent_and_scores_the_rubric(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    seen_messages = _patch_for_local_rollout(
        spec_module,
        monkeypatch,
        lab_task_root,
        body="The termination fee is $50,000 per notes.txt.",
    )
    case = Case(
        input={"task_id": "contracts/t1", "title": "task contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={
            "criteria": [
                {"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000.", "deliverables": []},
                {"id": "C2", "title": "Cites source", "match_criteria": "Cites notes.txt.", "deliverables": []},
            ]
        },
    )

    result = asyncio.run(
        spec_module._run_case(case=case, targets=_candidate_prompts(spec_module), runtime=None),
    )
    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    # The candidate prompts reached the model, not the seeds.
    prompt_text = "\n".join(str(getattr(message, "content", "")) for turn in seen_messages for message in turn)
    assert SYSTEM_MARKER in prompt_text
    assert TEMPLATE_MARKER in prompt_text

    assert result.output["deliverables_produced"] == ["memo.md"]
    assert result.output["finished"] is True
    assert score.field_scores == {ALL_PASS_FIELD: 1.0, CRITERION_PASS_RATE_FIELD: 1.0}
    assert score.objective == 1.0
    trace = result.run_metrics["trace_evidence"]
    assert set(trace["per_prompt_feedback"]) == {"system_prompt", "task_template"}


def test_run_case_reports_missing_deliverables_and_failed_criteria(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    _patch_for_local_rollout(spec_module, monkeypatch, lab_task_root, body="No numbers here.")
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={
            "criteria": [{"id": "C1", "title": "States the fee", "match_criteria": "Mentions $50,000."}],
        },
    )

    result = asyncio.run(spec_module._run_case(case=case, targets=_candidate_prompts(spec_module), runtime=None))
    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    assert score.objective == 0.0
    assert score.field_scores[ALL_PASS_FIELD] == 0.0
    feedback = result.run_metrics["trace_evidence"]["per_prompt_feedback"]["task_template"]
    assert any("failed criteria" in note for note in feedback)


def _failing_case() -> Case:
    return Case(
        input={
            "task_id": "contracts/t1",
            "instructions": "Summarize the fee.",
            "documents": ["notes.txt"],
        },
        case_id="contracts/t1",
        ground_truth={"criteria": [{"id": "C1", "match_criteria": "Mentions $50,000."}]},
    )


def test_a_harness_failure_is_reported_as_a_failed_rollout(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken harness must not look like an agent that scored 0.

    This is the regression for the run where a non-concurrency-safe task
    fetcher emptied every task folder: every rollout crashed, the spec turned
    each crash into an ordinary result, and 297 cases reported a clean 0.0.
    """

    def _boom(_case: Case, **_kwargs: Any) -> Any:
        raise FileNotFoundError("task fetch failed")

    monkeypatch.setattr(spec_module, "_record_for_case", _boom)

    result = asyncio.run(
        spec_module._run_case(case=_failing_case(), targets=_candidate_prompts(spec_module), runtime=None)
    )

    assert "task fetch failed" in result.output["error"]


def test_a_harness_failure_carries_the_sdk_failure_marker(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker is what makes the case unresolved instead of a scored zero."""

    def _boom(_case: Case, **_kwargs: Any) -> Any:
        raise FileNotFoundError("task fetch failed")

    monkeypatch.setattr(spec_module, "_record_for_case", _boom)

    result = asyncio.run(
        spec_module._run_case(case=_failing_case(), targets=_candidate_prompts(spec_module), runtime=None)
    )

    assert result.failure is not None
    assert result.failure.retryable is True
    assert "task fetch failed" in result.failure.error


def test_a_task_download_failure_is_a_harness_failure(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetcher reports GitHub rate limits as a plain ``RuntimeError``."""

    def _throttled(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("GitHub API rate limit exceeded")

    monkeypatch.setattr(spec_module, "ensure_task_dirs", _throttled)

    result = asyncio.run(
        spec_module._run_case(case=_failing_case(), targets=_candidate_prompts(spec_module), runtime=None)
    )

    assert "rate limit" in result.output["error"]


def test_a_stale_dataset_row_is_reported_as_a_permanent_failure(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying a fingerprint mismatch thirty times just hides it."""

    def _stale(_case: Case, **_kwargs: Any) -> Any:
        raise spec_module.DatasetMismatchError("contracts/t1: fetched task tree does not match the dataset row")

    monkeypatch.setattr(spec_module, "_record_for_case", _stale)

    result = asyncio.run(
        spec_module._run_case(case=_failing_case(), targets=_candidate_prompts(spec_module), runtime=None)
    )

    assert result.failure is not None
    assert result.failure.retryable is False


def test_a_provider_api_error_is_a_harness_failure(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiteLLM maps provider failures onto ``openai.*`` types, not its own base."""
    from litellm.exceptions import RateLimitError

    def _boom(_case: Case, **_kwargs: Any) -> Any:
        raise RateLimitError("slow down", llm_provider="openrouter", model="m")

    monkeypatch.setattr(spec_module, "_record_for_case", _boom)

    result = asyncio.run(
        spec_module._run_case(case=_failing_case(), targets=_candidate_prompts(spec_module), runtime=None)
    )

    assert "RateLimitError" in result.output["error"]


def test_an_unexpected_error_escapes_to_the_runtime(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only classified harness failures are caught; a bug must not be swallowed."""

    def _bug(_case: Case, **_kwargs: Any) -> Any:
        raise ValueError("this spec has a bug")

    monkeypatch.setattr(spec_module, "_record_for_case", _bug)

    with pytest.raises(ValueError, match="this spec has a bug"):
        asyncio.run(spec_module._run_case(case=_failing_case(), targets=_candidate_prompts(spec_module), runtime=None))


def test_a_task_with_no_documents_of_its_own_is_not_a_harness_failure(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
    tmp_path: Path,
) -> None:
    """Completeness is measured against the row, not against "documents/ is non-empty"."""
    for document in (lab_task_root / "contracts/t1/documents").iterdir():
        document.unlink()
    monkeypatch.setattr(spec_module, "ensure_task_dirs", lambda *_a, **_kw: lab_task_root)
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Draft from scratch.", "documents": []},
        case_id="contracts/t1",
        ground_truth={"criteria": [{"id": "C1", "match_criteria": "Mentions $50,000."}]},
    )

    record, _root = spec_module._record_for_case(case, cache_dir=tmp_path)

    assert record.task_id == "contracts/t1"


@pytest.mark.parametrize("missing", ["task.json", "documents"])
def test_an_empty_task_directory_fails_before_the_agent_runs(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
    tmp_path: Path,
    missing: str,
) -> None:
    """The one-stat precondition that would have caught the whole bad run."""
    task_dir = lab_task_root / "contracts" / "t1"
    if missing == "task.json":
        (task_dir / "task.json").unlink()
    else:
        for document in (task_dir / "documents").iterdir():
            document.unlink()

    monkeypatch.setattr(spec_module, "ensure_task_dirs", lambda *_a, **_kw: lab_task_root)

    with pytest.raises(spec_module.HarnessError, match="contracts/t1"):
        spec_module._record_for_case(_failing_case(), cache_dir=tmp_path)


def test_an_agent_that_submits_nothing_still_scores_an_honest_zero(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    """LAB-AA grades an empty submission as 0 — it is not a harness failure."""
    _patch_for_local_rollout(spec_module, monkeypatch, lab_task_root, body="unused", submit_nothing=True)
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={"criteria": [{"id": "C1", "match_criteria": "Mentions $50,000.", "deliverables": ["memo.md"]}]},
    )

    result = asyncio.run(spec_module._run_case(case=case, targets=_candidate_prompts(spec_module), runtime=None))
    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    assert getattr(result, "failure", None) is None, "an empty submission is a score, not a harness failure"
    assert result.output["deliverables_produced"] == []
    assert score.field_scores == {ALL_PASS_FIELD: 0.0, CRITERION_PASS_RATE_FIELD: 0.0}


def test_a_judge_that_never_answers_is_a_harness_failure_not_a_zero(spec_module: ModuleType) -> None:
    """``score_rubric`` scores an unreachable judge FAIL; that is not the agent's score."""

    def _dead(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
        raise spec_module.JudgeCallError("judge API is down")

    judge = spec_module._OutageAwareJudge(_dead)
    graded = spec_module.score_rubric(
        criteria=[{"id": "C1", "match_criteria": "Mentions $50,000.", "deliverables": ["memo.md"]}],
        deliverables={"memo.md": "The fee is $50,000."},
        task_description="Summarize the fee.",
        judge=judge,
    )

    assert graded[CRITERION_PASS_RATE_FIELD] == 0.0
    with pytest.raises(spec_module.JudgeCallError, match="nothing was graded"):
        judge.assert_reached()


def test_a_judge_that_answers_is_not_reported_as_an_outage(spec_module: ModuleType) -> None:
    """A criterion whose deliverables are absent never reaches the judge — still a score."""
    judge = spec_module._OutageAwareJudge(lambda *_a, **_kw: {"C1": False})
    spec_module.score_rubric(
        criteria=[{"id": "C1", "match_criteria": "Mentions $50,000.", "deliverables": ["memo.md"]}],
        deliverables={"memo.md": "Nothing relevant."},
        task_description="Summarize the fee.",
        judge=judge,
    )

    judge.assert_reached()


class _FakeCall:
    """The one trace operation a model call uses."""

    def __init__(self) -> None:
        self.recorded_usage: dict[str, int] | None = None
        self.output_value: Any = None

    def __enter__(self) -> "_FakeCall":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def usage(self, **kwargs: int) -> None:
        self.recorded_usage = dict(kwargs)

    def output(self, value: Any) -> None:
        self.output_value = value


class _FakeTrace:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], _FakeCall]] = []

    def model_call(self, **kwargs: Any) -> _FakeCall:
        call = _FakeCall()
        self.calls.append((kwargs, call))
        return call


class _UsageClient:
    """A Stirrup client whose reply carries the provider's token counts."""

    model_slug = "openrouter/some-model"
    max_tokens = 4096

    def __init__(self, *, input_tokens: int, answer: int, reasoning: int) -> None:
        self._usage = (input_tokens, answer, reasoning)

    async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
        from stirrup.core.models import AssistantMessage, TokenUsage

        input_tokens, answer, reasoning = self._usage
        return AssistantMessage(
            content="done",
            token_usage=TokenUsage(input=input_tokens, answer=answer, reasoning=reasoning),
        )


def test_a_traced_model_call_reports_the_providers_own_token_counts(spec_module: ModuleType) -> None:
    """The runtime derives rollout cost from these spans; nothing is estimated."""
    trace = _FakeTrace()
    client = spec_module._TracedClient(
        _UsageClient(input_tokens=120, answer=30, reasoning=8), trace=trace, provider="openrouter"
    )

    reply = asyncio.run(client.generate([], {}))

    assert reply.content == "done"
    (attributes, call) = trace.calls[0]
    assert attributes["model"] == "openrouter/some-model"
    assert attributes["provider"] == "openrouter"
    # Stirrup counts reasoning separately; the provider's output is both.
    assert call.recorded_usage == {"input_tokens": 120, "output_tokens": 38, "total_tokens": 158}


def test_a_traced_call_records_only_what_the_turn_added(spec_module: ModuleType) -> None:
    """Stirrup resends the whole history each turn; recording it all is quadratic."""
    trace = _FakeTrace()
    client = spec_module._TracedClient(
        _UsageClient(input_tokens=1, answer=1, reasoning=0), trace=trace, provider="openrouter"
    )
    history = ["system", "task"]

    asyncio.run(client.generate(list(history), {}))
    history.append("tool result")
    asyncio.run(client.generate(list(history), {}))
    asyncio.run(client.generate(["compacted"], {}))

    assert [attributes["input_messages"] for attributes, _call in trace.calls] == [
        ["system", "task"],
        ["tool result"],
        ["compacted"],
    ]


def test_a_model_call_without_reported_usage_records_none(spec_module: ModuleType) -> None:
    trace = _FakeTrace()
    client = spec_module._TracedClient(
        _UsageClient(input_tokens=0, answer=0, reasoning=0), trace=trace, provider="openrouter"
    )

    asyncio.run(client.generate([], {}))

    assert trace.calls[0][1].recorded_usage is None, "usage the provider never reported must not be invented"


def test_the_spec_traces_the_recipes_own_client(spec_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is a wrapper around the recipe's default factory, not a fork of it."""
    inner = _UsageClient(input_tokens=1, answer=1, reasoning=0)
    monkeypatch.setattr(spec_module, "_default_model_factory", lambda *_args, **_kwargs: inner)

    routing = spec_module._TaskModelRouting()
    client = spec_module._rollout_model_factory(_FakeTrace(), "openrouter", routing)("m", 0.0, 1, 1, 1.0, "none")

    assert isinstance(client, spec_module._TracedClient)
    assert client.model_slug == inner.model_slug


def test_a_hosted_rollout_routes_the_agent_through_the_inference_gateway(
    spec_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected model is reached through the run's gateway, not the provider.

    ``runtime.model`` is a bare model id, so LiteLLM cannot resolve a provider
    from it and the sandbox holds no provider key anyway. The gateway's
    canonical id and credentials must reach the client instead.
    """
    monkeypatch.setenv("BEAKER_INFERENCE_BASE_URL", "https://runtime.example/v1/llm")
    monkeypatch.setenv("BEAKER_INFERENCE_API_KEY", "run-token")
    runtime = RolloutContext(
        model="z-ai/glm-5.2",
        provider="openrouter",
        canonical_model_id="openrouter:z-ai/glm-5.2",
        user_id="u",
    )

    config, routing = spec_module._config_for_runtime(runtime)

    assert config.task_model == "openai/openrouter:z-ai/glm-5.2"
    # Neither the recipe's reasoning effort nor its DeepSeek-sized output cap
    # can be asserted against a model the recipe knows nothing about.
    assert config.task_reasoning_effort == "none"
    assert config.max_output_tokens == 16_384
    # The selected model's real window, so stirrup's summarization check is
    # sized against the context and not the output cap.
    assert config.context_window_tokens == 1_048_576
    assert routing.factory_kwargs() == {
        "api_base": "https://runtime.example/v1/llm",
        "api_key": "run-token",
    }

    seen: dict[str, Any] = {}

    def _record(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return _UsageClient(input_tokens=1, answer=1, reasoning=0)

    monkeypatch.setattr(spec_module, "_default_model_factory", _record)
    spec_module._rollout_model_factory(None, "openrouter", routing)(config.task_model, 0.0, 1, 1, 1.0, "none")

    assert seen == routing.factory_kwargs()


def test_a_run_without_a_selected_model_keeps_the_recipes_own_provider(spec_module: ModuleType) -> None:
    """Prompt-only optimization must not re-route the agent to another model."""
    config, routing = spec_module._config_for_runtime(RolloutContext(model=None, user_id="u"))

    assert config == HarveyLabConfig()
    assert routing.factory_kwargs() == {}


def test_targets_that_miss_the_agent_are_rejected(spec_module: ModuleType, lab_task_root: Path) -> None:
    """The guard that keeps the optimizer from tuning prompts the agent ignores."""
    system_prompt, task_template = load_harvey_lab_prompts()
    agent = HarveyLabAgent(
        config=HarveyLabConfig(max_turns=5, enable_view_image=False),
        task_source=task_source_from_dir(lab_task_root),
        system_prompt=system_prompt,
        task_template=task_template,
    )
    spec_module._assert_targets_applied(agent, {"system_prompt": system_prompt, "task_template": task_template})
    with pytest.raises(RuntimeError, match="task_template"):
        spec_module._assert_targets_applied(agent, {"system_prompt": system_prompt, "task_template": "something else"})
    with pytest.raises(RuntimeError, match="system_prompt"):
        spec_module._assert_targets_applied(
            agent, {"system_prompt": "another system prompt", "task_template": task_template}
        )


def test_record_for_case_rejects_a_task_tree_that_drifted_from_the_row(
    spec_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    lab_task_root: Path,
) -> None:
    monkeypatch.setattr(spec_module, "ensure_task_dirs", lambda *_a, **_kw: lab_task_root)
    case = Case(
        input={"task_id": "contracts/t1", "instructions": "Summarize the fee."},
        case_id="contracts/t1",
        ground_truth={"criteria": []},
        metadata={"task_fingerprint": "deadbeef" * 8},
    )
    with pytest.raises(RuntimeError, match="does not match the dataset row"):
        spec_module._record_for_case(case)


# ─── scoring ──────────────────────────────────────────────────────────


def test_scorer_counts_only_verdicts_for_the_rows_own_criteria(spec_module: ModuleType) -> None:
    case = Case(
        input={},
        case_id="c",
        ground_truth={
            "criteria": [
                {"id": "C1", "match_criteria": "a"},
                {"id": "C2", "match_criteria": "b"},
                {"id": "C3", "match_criteria": "c"},
                {"id": "C4", "match_criteria": "d"},
            ]
        },
    )
    result = CaseResult(
        output={},
        context={
            "verdicts": [
                {"id": "C1", "passed": True},
                {"id": "C2", "passed": False},
                {"id": "C3", "passed": True},
                # A verdict for a criterion this row does not carry is ignored.
                {"id": "C9", "passed": True},
            ]
        },
    )

    score = asyncio.run(spec_module.RubricScorer().score_case(case=case, result=result))

    assert score.field_scores == {ALL_PASS_FIELD: 0.0, CRITERION_PASS_RATE_FIELD: 0.5}
    assert score.objective == 0.5


# ─── spec assembly ────────────────────────────────────────────────────


def test_build_spec_is_valid_and_declares_its_dataset_schema(spec_module: ModuleType) -> None:
    built = validate_spec(spec_module.build_spec(OptimizationContext()))
    assert built.data_loader.dataset_schema is spec_module.DATASET_SCHEMA
    registration = spec_module.build_spec.__beaker_spec__
    assert registration.metadata["dataset_schema"]["json_schema"]["required"] == ["id", "input", "expected"]


def test_a_hosted_run_without_a_judge_credential_fails_at_spec_load(
    spec_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cheaper to fail here than after every rollout has run the agent to completion."""
    monkeypatch.setenv("BEAKER_INFERENCE_BASE_URL", "https://runtime.example/v1/llm")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        spec_module.build_spec(OptimizationContext())

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert validate_spec(spec_module.build_spec(OptimizationContext())) is not None
