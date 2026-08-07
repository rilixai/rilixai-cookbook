"""Beaker prompt-optimization spec for the Harvey LAB legal agent.

What is optimized: the recipe's **two agent prompts** — Artificial Analysis'
LAB-AA ``system_prompt`` and ``task_template`` (``harvey_lab.agent.prompts``).
They are the only text the agent is steered with, and ``HarveyLabAgent`` already
takes both as constructor overrides, so a candidate bundle reaches the real
agent without touching recipe code.

One case = one Harvey LAB task:

* ``input`` — the task id, title, instructions and requested deliverable
  filenames (see ``../.beaker/export_dataset.py``).
* ``ground_truth`` — that task's own rubric: ~40-70 binary ``match_criteria``.
* rollout — the task folder is fetched from ``harveyai/harvey-labs`` at the
  pinned commit, then the Stirrup agent runs it under the candidate prompts
  with its single ``code_exec`` tool and submits deliverables through ``finish``.
* score — the recipe's batched rubric judge grades every criterion, and the
  scorer aggregates those verdicts into the two LAB metrics: ``all_pass``
  (1.0 iff every criterion passed) and ``criterion_pass_rate`` (the share that
  passed). The optimizer maximizes ``criterion_pass_rate``, because all-pass is
  far too sparse a signal on a 60-criterion task to steer a search.

Grading runs inside ``run_case`` (as ``harvey-lab evaluate`` does) so the
per-criterion verdicts can ride along as reflection evidence; ``score_case``
stays deterministic and only aggregates them.

This module lives at the recipe root rather than beside ``beaker.yaml`` at the
repository root, because ``spec.source_dir`` points the image builder at
``harvey_lab/`` so it pip-installs this recipe's ``pyproject.toml``. Only what
is under ``source_dir`` reaches the image. See the note in
``.beaker/beaker.yaml``.

Temporary contortions
---------------------
Six things here are shaped by a platform gap rather than by the benchmark, and
each should be deleted once the gap closes. They are listed together because
individually they read like ordinary code; together they are the standing bill
for running LAB-AA on Beaker. Each is explained where it lives.

1. **Reasoning is off in a hosted run** — ``_config_for_runtime``. The gateway
   validates ``reasoning_effort`` against ``model_catalog``, which marks both
   GLM 5.2 and DeepSeek V4 Pro non-reasoning, so the request is rejected before
   it leaves us. AA runs reasoning models at maximum effort, so *hosted numbers
   are not comparable to published LAB results until this is fixed.* Needs the
   catalog flags corrected.
2. **Output is capped at 16,384 tokens in a hosted run** — ``_config_for_runtime``.
   AA gives a reasoning model its creator's maximum, which the recipe knows for
   its own model and cannot know for a selected one. Needs the selected model's
   capabilities (max output, context window, reasoning tier) surfaced to the
   rollout; that single addition removes this, (1), and the
   ``_CONTEXT_WINDOW_TOKENS`` table beside it.
3. **The judge needs its own provider key in the sandbox** —
   ``_assert_judge_credential_present``. Needs the gateway to authorize a
   run-scoped grader model alongside the model targets.
4. **Judge cost is unmeasured** — ``_grade``. Only the agent's calls are traced.
5. **The recipe's dependencies are declared to the image by pointing
   ``source_dir`` at this directory** — ``.beaker/beaker.yaml``. Needs the
   ``pip_install_from`` key that ``beaker init`` already writes and nothing
   reads.
6. **Agent binaries are hand-listed as ``apt_install``** — same file. No
   packaging metadata can express them, so the list can drift from what the
   agent shells out to; ``_warn_on_missing_agent_binaries`` is the backstop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseDataLoader,
    CaseResult,
    CaseScore,
    DatasetRowContext,
    DatasetSchema,
    OptimizationContext,
    OptimizationTargets,
    Spec,
    objective_score,
    optimization_targets_from_prompts,
    spec,
)
from beaker.sdk import RolloutContext, inference_target
from beaker.tracing import Trace

from harvey_lab.agent.agent import HarveyLabAgent, HarveyLabAgentOutput, _default_model_factory
from harvey_lab.agent.workspace import task_source_from_dir
from harvey_lab.config import HARVEY_LABS_COMMIT, HarveyLabConfig
from harvey_lab.data.dataset import HarveyLabRecord, load_records
from harvey_lab.data.fetch import ensure_task_dirs
from harvey_lab.evaluation.scoring import (
    ALL_PASS_FIELD,
    CRITERION_PASS_RATE_FIELD,
    JudgeCallError,
    build_rubric_judge,
    score_rubric,
)


logger = logging.getLogger(__name__)

# The prompt targets, named after the ``HarveyLabAgent`` keyword arguments they
# are threaded into.
SYSTEM_PROMPT_TARGET = "system_prompt"
TASK_TEMPLATE_TARGET = "task_template"

# Scored fields: Harvey LAB's two headline metrics.
FIELD_NAMES = (ALL_PASS_FIELD, CRITERION_PASS_RATE_FIELD)
# all_pass is reported per case but carries no weight in the objective: on a
# ~60-criterion rubric it is 0 for almost every candidate, so optimizing it
# directly gives the search no gradient.
OBJECTIVE_WEIGHTS = {CRITERION_PASS_RATE_FIELD: 1.0}

# Deliverable text is far too large to ship in the trajectory; reflection only
# needs enough of it to see what the agent actually produced.
_PREVIEW_CHARS = 800


class HarnessError(RuntimeError):
    """The harness failed, so this case never got a fair run.

    Kept distinct from every other exception on purpose. Artificial Analysis
    draws the same line: an agent that fails the task scores 0, while a run
    with persistent infrastructure failures is retried and, if it keeps
    failing, excluded rather than published. Collapsing the two is how a run
    where every rollout crashed reported a clean ``0.0`` for 297 cases.
    """


class DatasetMismatchError(HarnessError):
    """The dataset row and the fetched task disagree — retrying cannot help.

    Still a harness failure (the case never got a fair run), but a permanent
    one: the export is stale, and thirty identical retries would end in the
    case being quietly excluded instead of an operator re-exporting it.
    Reported with ``retryable=False`` so it stays a visible configuration bug.
    """


def _harness_error_types() -> tuple[type[BaseException], ...]:
    """Exception types that mean "the harness broke", not "the agent was wrong".

    Deliberately enumerated instead of catching bare ``Exception``: anything
    not listed here is a bug in this spec or the recipe, and it must escape to
    Beaker, which logs the traceback and records the case as unresolved. Both
    routes are loud; only these are reported as *retryable*.

    Covers the task tree (:class:`HarnessError`), any filesystem/network error
    (``OSError`` — the fetcher's ``urllib`` errors included), timeouts, every
    provider API error, Stirrup running out of context, and a rubric judge that
    never returned verdicts.

    The provider base is ``openai.OpenAIError``, not LiteLLM's same-named
    subclass: LiteLLM maps each provider failure onto the matching ``openai.*``
    type (``litellm.exceptions.RateLimitError`` derives from
    ``openai.RateLimitError``), so only the openai base catches them all.
    """
    types: list[type[BaseException]] = [HarnessError, OSError, TimeoutError, JudgeCallError]
    try:
        from openai import OpenAIError
    except Exception:  # noqa: BLE001 - optional at import time; only narrows the catch
        logger.warning("openai exceptions are unavailable; provider API errors will surface as unresolved cases.")
    else:
        types.append(OpenAIError)
    try:
        from stirrup.core.exceptions import ContextOverflowError
    except Exception:  # noqa: BLE001 - optional at import time; only narrows the catch
        logger.warning("stirrup exceptions are unavailable; context overflows will surface as unresolved cases.")
    else:
        types.append(ContextOverflowError)
    return tuple(types)


HARNESS_ERRORS = _harness_error_types()


# ── Dataset loading ──────────────────────────────────────────────────────────

DATASET_SCHEMA = DatasetSchema(
    json_schema={
        "type": "object",
        "required": ["id", "input", "expected"],
        "properties": {
            "id": {"type": "string", "minLength": 1, "description": "LAB task id (path under the repo's tasks/)."},
            "input": {
                "type": "object",
                "required": ["task_id", "instructions"],
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "title": {"type": "string"},
                    "work_type": {"type": "string"},
                    "instructions": {"type": "string", "minLength": 1},
                    "deliverables": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Requested output filename -> canonical name; graded by exact filename.",
                    },
                    "documents": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            "expected": {
                "type": "object",
                "required": ["criteria"],
                "properties": {
                    "criteria": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["id", "match_criteria"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "title": {"type": "string"},
                                "match_criteria": {"type": "string", "minLength": 1},
                                "deliverables": {"type": "array", "items": {"type": "string"}},
                            },
                            "additionalProperties": True,
                        },
                    }
                },
                "additionalProperties": True,
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "practice_area": {"type": "string"},
                    "harvey_labs_commit": {"type": "string"},
                    "task_fingerprint": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "group_key": {"type": "string", "description": "Practice area, so scores group the way LAB is drawn."},
        },
        "additionalProperties": True,
    }
)


@dataclass(frozen=True)
class LabTaskRow:
    """One validated JSONL row: a LAB task plus its rubric."""

    task_id: str
    title: str
    work_type: str
    instructions: str
    deliverables: dict[str, str]
    documents: tuple[str, ...]
    criteria: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def practice_area(self) -> str:
        return str(self.metadata.get("practice_area") or self.task_id.split("/", 1)[0])


class LabTaskDataLoader(CaseDataLoader[LabTaskRow]):
    """Validate ``.beaker/dataset/*.jsonl`` rows and map them to cases.

    Layout (``config_defaults.local_dataset_path`` for dry-runs, an uploaded
    dataset revision for hosted runs)::

        .beaker/dataset/train.jsonl      # required
        .beaker/dataset/val.jsonl        # required
        .beaker/dataset/test.jsonl       # optional
        .beaker/dataset/manifest.json    # provenance: source repo + commit

    Rows are produced by ``.beaker/export_dataset.py`` from the frozen
    ``harvey_lab`` splits; a row without a scoreable criterion is rejected here
    rather than silently scoring 0 (the recipe treats such a task as
    unscoreable and excludes it from its own averages).
    """

    dataset_schema = DATASET_SCHEMA

    def parse_row(self, raw: Mapping[str, Any], context: DatasetRowContext) -> LabTaskRow:
        del context
        payload = _require_mapping(raw.get("input"), "input")
        expected = _require_mapping(raw.get("expected"), "expected")
        metadata = _require_mapping(raw.get("metadata") or {}, "metadata")
        task_id = str(raw.get("id") or payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("id must be a non-empty LAB task id")
        instructions = str(payload.get("instructions") or "").strip()
        if not instructions:
            raise ValueError(f"{task_id}: input.instructions must be non-empty")
        criteria = _parse_criteria(expected.get("criteria"), task_id)
        deliverables = _require_mapping(payload.get("deliverables") or {}, "input.deliverables")
        return LabTaskRow(
            task_id=task_id,
            title=str(payload.get("title") or ""),
            work_type=str(payload.get("work_type") or ""),
            instructions=instructions,
            deliverables={str(k): str(v) for k, v in deliverables.items()},
            documents=tuple(str(name) for name in payload.get("documents") or ()),
            criteria=criteria,
            metadata=dict(metadata),
        )

    def iter_cases(self, row: LabTaskRow, context: DatasetRowContext) -> Iterable[Case]:
        del context
        _preflight_task(row.task_id, str(row.metadata.get("harvey_labs_commit") or HARVEY_LABS_COMMIT))
        yield Case(
            input={
                "task_id": row.task_id,
                "title": row.title,
                "work_type": row.work_type,
                "instructions": row.instructions,
                "deliverables": dict(row.deliverables),
                "documents": list(row.documents),
            },
            case_id=row.task_id,
            ground_truth={"criteria": [dict(criterion) for criterion in row.criteria]},
            group_key=row.practice_area,
            metadata=dict(row.metadata),
        )


def _preflight_task(task_id: str, commit: str) -> None:
    """Materialize a task's tree while the dataset is being loaded.

    This is the run's single pre-flight choke point: ``iter_cases`` runs for
    every row before the engine schedules the first rollout, so by the time any
    ``run_case`` starts, every task the run needs is already a complete tree in
    the cache and the rollout path does no downloading. That matters because
    fetching during rollouts means dozens of workers racing on the same task
    (one LAB task is a ~3k-file data room), and a task that lands *after* its
    case was evaluated scores 0 with no tool calls at all.

    Best-effort by design: a task that cannot be fetched here must not abort the
    whole run at dataset-load time, so it is logged and left to ``run_case``,
    where the failure is attributed to its own case. ``ensure_task_dirs`` is
    itself concurrency-safe, so that fallback is a slow path, not a race.
    """
    try:
        ensure_task_dirs([task_id], commit=commit)
    except Exception as exc:  # noqa: BLE001 - pre-flight is an optimization, run_case still guards
        logger.warning(
            "Pre-flight fetch of LAB task %s failed (%s: %s); retrying in run_case.",
            task_id,
            type(exc).__name__,
            exc,
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _parse_criteria(value: Any, task_id: str) -> tuple[dict[str, Any], ...]:
    """Validate the rubric: a non-empty list of binary criteria with standards."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{task_id}: expected.criteria must be a JSON array")
    criteria: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        entry_map = _require_mapping(entry, f"{task_id}: expected.criteria[{index}]")
        match_criteria = str(entry_map.get("match_criteria") or "").strip()
        if not match_criteria:
            raise ValueError(f"{task_id}: expected.criteria[{index}].match_criteria must be non-empty")
        criteria.append(
            {
                "id": str(entry_map.get("id") or f"C-{index + 1:03d}"),
                "title": str(entry_map.get("title") or ""),
                "match_criteria": match_criteria,
                "deliverables": [str(name) for name in entry_map.get("deliverables") or ()],
            }
        )
    if not criteria:
        raise ValueError(f"{task_id}: expected.criteria is empty, so the task is unscoreable")
    return tuple(criteria)


# ── Rollout ──────────────────────────────────────────────────────────────────


def _seed_targets() -> OptimizationTargets:
    """The prompts the agent runs on today, straight from the recipe."""
    from harvey_lab.agent.prompts import load_harvey_lab_prompts

    system_prompt, task_template = load_harvey_lab_prompts()
    return optimization_targets_from_prompts(
        {SYSTEM_PROMPT_TARGET: system_prompt, TASK_TEMPLATE_TARGET: task_template}
    )


# AA's LAB-AA protocol caps a non-reasoning model's completion at 16,384
# tokens; a reasoning model instead gets its creator's maximum, which is what
# ``HarveyLabConfig.max_output_tokens`` encodes for the model the recipe ships
# with.
_AA_NON_REASONING_OUTPUT_TOKENS = 16_384

# Stirrup 0.2 sizes its history-summarization check against the model's real
# context window, passed separately from the output cap above. The platform
# does not yet surface a selected model's window to the rollout, so the
# LAB-relevant OpenRouter models are listed here (per OpenRouter's model
# metadata) with a floor every model in the catalog clears. Lifts with the
# same capability-surfacing change as contortions 1 and 2.
_CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "qwen/qwen3.7-max": 1_000_000,
    "qwen/qwen3.7-plus": 1_000_000,
    "z-ai/glm-5.2": 1_048_576,
    "z-ai/glm-5v-turbo": 202_752,
    "z-ai/glm-4.7": 204_800,
    "z-ai/glm-4.7-flash": 202_752,
    "deepseek/deepseek-v4-pro": 1_048_576,
    "deepseek/deepseek-v4-flash": 1_048_576,
    "deepseek/deepseek-v3.2": 163_840,
}
_CONTEXT_WINDOW_FLOOR = 131_072


@dataclass(frozen=True)
class _TaskModelRouting:
    """Where the agent's own LLM calls go for one rollout.

    ``api_base``/``api_key`` are set only for a hosted run, where they address
    the run's inference gateway. Empty everywhere else, which leaves the recipe
    calling its configured provider directly, exactly as ``harvey-lab run`` does.
    """

    api_base: str | None = None
    api_key: str | None = None

    def factory_kwargs(self) -> dict[str, str]:
        kwargs: dict[str, str] = {}
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        return kwargs


def _config_for_runtime(runtime: RolloutContext | Any) -> tuple[HarveyLabConfig, _TaskModelRouting]:
    """The recipe's production config, re-pointed at the model Beaker selected.

    Without a selected model the recipe's own default stands: prompt-only
    optimization must not silently re-route the agent to another model.

    With one, note what ``runtime`` actually carries. ``runtime.model`` is a
    *bare* model id (``z-ai/glm-5.2``) and the provider is a separate field —
    it is **not** a LiteLLM model spec, and handing it to LiteLLM as one is a
    BadRequestError ("LLM Provider NOT provided") for any id whose first path
    segment LiteLLM does not recognise as a provider. The canonical id lives on
    ``runtime.canonical_model_id`` (``openrouter:z-ai/glm-5.2``) and is what the
    gateway wants.

    So a hosted rollout is routed through the run's inference gateway, which
    :func:`beaker.sdk.inference_target` resolves from the sandbox environment.
    That is the designed path, and it is load-bearing for more than provider
    resolution:

    * the sandbox holds no provider credential of its own — the gateway spends
      the customer's, against the run's budget;
    * the gateway is what attributes spend to *this* model target, which is
      where the model-comparison chart's per-target rollout cost comes from.
      Call OpenRouter directly and that cost is simply never recorded.

    The gateway speaks OpenAI's dialect, hence the ``openai/`` prefix: it tells
    LiteLLM which dialect to speak, while the model *named in the request body*
    stays the canonical id the gateway authorizes against.
    """
    config = HarveyLabConfig()
    selected_model = str(getattr(runtime, "model", None) or "")
    if not selected_model:
        return config, _TaskModelRouting()

    target = inference_target(runtime)
    return (
        replace(
            config,
            task_model=f"openai/{target.model}",
            # The recipe's default effort is tuned for the model it ships with.
            # A swapped-in model may not be a reasoning model at all, and the
            # gateway rejects the parameter outright for one whose catalog entry
            # says so, so the recipe cannot keep asserting its own default here.
            # Restore this once a rollout is told the selected model's tier
            # (contortion 1: AA runs reasoning models at maximum effort, so a
            # hosted score is not comparable to a published LAB one meanwhile).
            task_reasoning_effort="none",
            # Same reason, with teeth: the recipe's 384k cap is DeepSeek V4's
            # own maximum output, which is what AA's protocol asks of a
            # *reasoning* model. Asserted against another model it is at best
            # above that model's limit, and at worst unaffordable — OpenRouter
            # reserves ``max_tokens`` × the output rate against the account
            # balance before it will start a request, so 384k on GLM 5.2
            # (~$2.86/M out) demands roughly $1.1k of headroom per call and is
            # refused with HTTP 402 whatever the balance actually is. Non-
            # reasoning models get AA's 16,384, which matches the effort we
            # just dropped. Contortion 2, and it lifts with the same change.
            max_output_tokens=_AA_NON_REASONING_OUTPUT_TOKENS,
            context_window_tokens=_CONTEXT_WINDOW_TOKENS.get(selected_model, _CONTEXT_WINDOW_FLOOR),
        ),
        _TaskModelRouting(api_base=target.base_url, api_key=target.api_key),
    )


def _record_for_case(case: Case, *, cache_dir: Path | None = None) -> tuple[HarveyLabRecord, Path]:
    """Fetch the case's LAB task folder at the pinned commit and load its record.

    The documents are the task, so they cannot live in the JSONL row: the row's
    ``task_id`` is resolved against ``harveyai/harvey-labs`` at
    ``metadata.harvey_labs_commit`` (cached, resumable) exactly like
    ``harvey-lab run``. The rubric that grades the rollout is the row's frozen
    copy, not the freshly read ``task.json``, so a benchmark edit can never move
    the labels underneath a dataset revision — a fingerprint mismatch fails the
    case instead.
    """
    payload = case.input if isinstance(case.input, Mapping) else {}
    task_id = str(payload.get("task_id") or case.case_id)
    commit = str(case.metadata.get("harvey_labs_commit") or HARVEY_LABS_COMMIT)
    try:
        tasks_root = ensure_task_dirs([task_id], commit=commit, cache_dir=cache_dir)
        _assert_task_workspace_usable(
            tasks_root / task_id,
            task_id,
            expected_documents=[str(name) for name in payload.get("documents") or ()],
        )
        record = load_records(tasks_root, task_ids=[task_id])[0]
    except HarnessError:
        raise
    except Exception as exc:
        # Everything between here and the agent is the harness fetching data.
        # The fetcher reports GitHub rate limits and exhausted retries as plain
        # ``RuntimeError``, and a truncated ``task.json`` surfaces as a
        # ``JSONDecodeError``; classified by type they would look like bugs in
        # this spec, when they are the retryable outage this PR exists to
        # report. Classified by *position* instead, they cannot be missed.
        raise HarnessError(f"{task_id}: could not resolve the task tree: {type(exc).__name__}: {exc}") from exc
    expected_fingerprint = str(case.metadata.get("task_fingerprint") or "")
    if expected_fingerprint and expected_fingerprint != record.task_fingerprint:
        raise DatasetMismatchError(
            f"{task_id}: fetched task tree does not match the dataset row "
            f"(fingerprint {record.task_fingerprint[:12]} != {expected_fingerprint[:12]}); "
            "re-export the dataset from the pinned commit."
        )
    return record, tasks_root


def _assert_task_workspace_usable(task_dir: Path, task_id: str, *, expected_documents: Sequence[str]) -> None:
    """Refuse to start the agent on a task folder the fetch did not complete.

    A few stat calls, before a single token is spent. A concurrent/interrupted
    fetch can leave a task directory that exists but holds no ``task.json`` or
    only some of its documents; the agent then dutifully explores an empty
    folder, submits nothing, and the run records an ordinary 0 for what is
    really a broken harness. This turns that into a loud, retryable failure in
    the first second of the rollout.

    The dataset row lists the documents it was exported with, so completeness
    is measured against *that* rather than against "``documents/`` is
    non-empty": a task the benchmark ships with no source documents is a task,
    not a failed download.
    """
    if not (task_dir / "task.json").is_file():
        raise HarnessError(f"{task_id}: no task.json under {task_dir} — the task tree was not fetched.")
    documents = task_dir / "documents"
    missing = [name for name in expected_documents if not (documents / name).is_file()]
    if missing:
        raise HarnessError(
            f"{task_id}: {len(missing)}/{len(expected_documents)} documents missing under {documents} "
            f"(first: {missing[0]}) — the task documents were not fetched."
        )


def _placeholder_free_fragment(prompt: str) -> str:
    """The longest chunk of ``prompt`` with no ``{{placeholder}}`` in it."""
    fragments = [chunk.split("}}")[-1] for chunk in prompt.split("{{")]
    return max(fragments, key=len).strip()


def _assert_targets_applied(agent: HarveyLabAgent, prompts: Mapping[str, str]) -> None:
    """Fail loudly if a candidate prompt never reached the agent.

    Cheap insurance against the classic silent failure: optimizing prompts the
    application does not actually use. ``task_template`` is stored verbatim and
    rendered per task; ``system_prompt`` is rendered once at construction
    (``{{max_turns}}`` and the tool names), so it is checked by its longest
    placeholder-free fragment.
    """
    if agent.task_template != prompts[TASK_TEMPLATE_TARGET]:
        raise RuntimeError("the task_template target did not reach HarveyLabAgent.")
    fragment = _placeholder_free_fragment(prompts[SYSTEM_PROMPT_TARGET])
    if fragment and fragment not in agent.system_prompt:
        raise RuntimeError("the system_prompt target did not reach HarveyLabAgent.")


def _task_description(case: Case, record: HarveyLabRecord) -> str:
    payload = case.input if isinstance(case.input, Mapping) else {}
    title = str(payload.get("title") or record.title).strip()
    header = f"{title}\n\n" if title else ""
    return f"{header}{record.instructions}".strip()


class _OutageAwareJudge:
    """A rubric judge that remembers whether it ever actually graded anything.

    ``score_rubric``'s ``_call_judge_with_fallback`` retries a batch and then
    scores it FAIL rather than aborting the evaluation — correct per criterion,
    but it means a judge API that is *entirely* down returns a full sheet of
    zeros that is indistinguishable from an agent that got everything wrong.
    That is the same silent-zero this PR exists to remove, one layer up, so the
    spec watches from outside: if the judge was called and no call ever
    succeeded, the case is a harness failure, not a score.

    A partial outage keeps LAB-AA's conservative behaviour untouched.
    """

    def __init__(self, judge: Any) -> None:
        self._judge = judge
        self.calls = 0
        self.verdicts = 0

    def __call__(self, task_description: str, criteria: Sequence[Mapping[str, Any]], agent_output: str) -> Any:
        self.calls += 1
        graded = self._judge(task_description, criteria, agent_output)
        self.verdicts += 1
        return graded

    def assert_reached(self) -> None:
        if self.calls and not self.verdicts:
            raise JudgeCallError(
                f"the rubric judge failed every one of its {self.calls} attempts; nothing was graded."
            )


def _grade(
    *,
    case: Case,
    output: HarveyLabAgentOutput,
    record: HarveyLabRecord,
    config: HarveyLabConfig,
) -> dict[str, Any]:
    """Grade the submitted deliverables against the row's frozen rubric.

    TODO (contortion 4): the judge's own token usage is not reported. Only the agent's calls
    are traced (see :class:`_TracedClient`), because the judge's LiteLLM calls
    are made inside ``harvey_lab.evaluation.scoring``; reaching their ``usage``
    blocks needs a change in that module, so grading cost is still unmeasured.
    """
    criteria = [dict(criterion) for criterion in case.ground_truth.get("criteria", ())]
    judge = _OutageAwareJudge(
        build_rubric_judge(
            config.judge_model,
            timeout=config.judge_llm_timeout,
            num_retries=config.judge_num_retries,
        )
    )
    graded = score_rubric(
        criteria=criteria,
        deliverables=output.deliverables,
        task_description=_task_description(case, record),
        judge=judge,
        batch_size=config.judge_batch_size,
    )
    judge.assert_reached()
    return graded


class _TracedClient:
    """A Stirrup ``LLMClient`` that reports each call to Beaker's trace.

    The recipe talks to the provider directly rather than through Beaker's
    gateway, so the runtime can only see what the run reports. Wrapping the
    client is the whole integration: one ``model_call`` span per request,
    carrying the provider's own token counts off Stirrup's ``TokenUsage``, from
    which the runtime derives ``outer_agent_usage`` itself. Nothing is
    estimated, and nothing here interprets the call.
    """

    def __init__(self, inner: Any, *, trace: Trace, provider: str) -> None:
        self._inner = inner
        self._trace = trace
        self._provider = provider
        self._recorded_messages = 0

    @property
    def model_slug(self) -> str:
        return str(self._inner.model_slug)

    @property
    def max_tokens(self) -> int:
        return int(self._inner.max_tokens)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _unrecorded(self, messages: list[Any]) -> list[Any]:
        """The messages this span adds to the conversation, not the whole of it.

        Stirrup passes the accumulated history on every turn, so recording all
        of it per call is quadratic in turns — and with ``max_turns=200`` and
        image content in the history, that is hundreds of megabytes of the same
        bytes. The spans concatenate to the full conversation anyway. A history
        shorter than what was already recorded means it was compacted (ReSum),
        so that one is recorded whole.
        """
        new = messages[self._recorded_messages :] if len(messages) >= self._recorded_messages else list(messages)
        self._recorded_messages = len(messages)
        return [_traceable_message(message) for message in new]

    async def generate(self, messages: list[Any], tools: dict[str, Any]) -> Any:
        with self._trace.model_call(
            operation="chat",
            provider=self._provider,
            model=self.model_slug,
            input_messages=self._unrecorded(messages),
        ) as call:
            reply = await self._inner.generate(messages, tools)
            usage = getattr(reply, "token_usage", None)
            if usage is not None:
                # Stirrup's terminology: output = answer + reasoning.
                prompt_tokens = int(usage.input)
                completion_tokens = int(usage.answer) + int(usage.reasoning)
                if prompt_tokens or completion_tokens:
                    call.usage(
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
            blocks = getattr(reply, "blocks", None)
            if blocks is not None:
                from stirrup.core.models import joined_text

                call.output(joined_text(blocks))
            else:
                call.output(str(getattr(reply, "content", "") or ""))
            return reply


def _traceable_message(message: Any) -> Any:
    """A trace-serializable view of a Stirrup chat message."""
    dump = getattr(message, "model_dump", None)
    return dump(mode="json") if callable(dump) else message


def _rollout_model_factory(trace: Trace | None, provider: str, routing: _TaskModelRouting) -> Any:
    """The recipe's own model factory, routed for this rollout and traced.

    ``_default_model_factory`` is private to ``harvey_lab.agent.agent``, and
    the spec reaches for it deliberately: the wrapper must build *exactly* the
    client the recipe would have built, so neither the routing nor the tracing
    changes anything about how the agent runs. Renaming it would break this
    import — export it publicly if that becomes a concern.
    """
    routing_kwargs = routing.factory_kwargs()

    def _factory(
        model: str,
        temperature: float,
        max_tokens: int,
        context_window_tokens: int,
        timeout: float,
        reasoning_effort: str,
    ) -> Any:
        client = _default_model_factory(
            model, temperature, max_tokens, context_window_tokens, timeout, reasoning_effort, **routing_kwargs
        )
        if trace is None:
            return client
        return _TracedClient(client, trace=trace, provider=provider)

    return _factory


def _trace_evidence(
    *,
    case: Case,
    output: HarveyLabAgentOutput,
    graded: Mapping[str, Any],
    criteria: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact per-case evidence the optimizer reflects on.

    The useful signal for a prompt rewrite is *which* rubric criteria the work
    product missed and whether the agent even submitted the requested files, so
    those are attributed to the prompt that governs them: submission mechanics
    to the system prompt, task/deliverable framing to the task template.
    """
    titles = {str(criterion.get("id")): str(criterion.get("title") or "") for criterion in criteria}
    failed = [
        f"{verdict.get('id')}: {titles.get(str(verdict.get('id')), '')}".strip()
        for verdict in graded.get("verdicts", ())
        if not verdict.get("passed")
    ]
    submission_notes = [
        f"finished={output.finished} abandoned={output.abandoned} "
        f"max_turns_reached={output.max_turns_reached} turns={output.total_turns}"
    ]
    if output.missing_deliverables:
        submission_notes.append(
            f"never submitted through finish: {', '.join(sorted(output.missing_deliverables))} "
            "(a missing file fails every criterion scoped to it)"
        )
    rubric_notes = [f"{graded.get('passed')}/{graded.get('total_criteria')} rubric criteria passed"]
    if failed:
        rubric_notes.append("failed criteria: " + "; ".join(failed[:25]))
    return {
        "case_id": case.case_id,
        "input_summary": _task_description(case, _EMPTY_RECORD)[:500],
        "per_prompt_feedback": {
            SYSTEM_PROMPT_TARGET: submission_notes,
            TASK_TEMPLATE_TARGET: rubric_notes,
        },
        "application_notes": [
            f"deliverables produced: {', '.join(sorted(output.deliverables)) or '(none)'}",
            f"agent wall time: {output.wall_seconds:.1f}s",
        ],
    }


# Placeholder record used only to render an input summary from the case payload.
_EMPTY_RECORD = HarveyLabRecord(
    task_id="",
    practice_area="",
    title="",
    work_type="",
    instructions="",
    deliverables={},
    criteria=(),
    documents=(),
    raw_task={},
    task_fingerprint="",
)


async def _run_case(*, case: Case, targets: OptimizationTargets, runtime: Any) -> CaseResult:
    """Run one LAB task under the candidate prompts, then grade it.

    Failure policy, mirroring AA's: a harness failure (the task tree missing or
    empty, the sandbox or an API dying, grading never completing) is reported
    as a *failed* rollout — ``CaseResult.failed`` carries a ``CaseFailure`` the
    runtime counts as unresolved, so a systematically broken run cannot be
    mistaken for a bad score. An agent that runs to completion and submits
    nothing is NOT a harness failure: it flows through to the scorer and earns
    an honest 0, exactly as LAB-AA grades an empty submission.
    """
    prompts = targets.to_dict()
    config, routing = _config_for_runtime(runtime)
    trace = getattr(runtime, "trace", None)
    try:
        record, tasks_root = await asyncio.to_thread(_record_for_case, case)
        agent_kwargs: dict[str, Any] = {}
        if trace is not None or routing.factory_kwargs():
            # Only when the runtime traces the rollout or selects where its
            # calls go, so the recipe's own default model factory stays in
            # charge everywhere else.
            agent_kwargs["model_factory"] = _rollout_model_factory(
                trace, str(getattr(runtime, "provider", None) or "litellm"), routing
            )
        agent = HarveyLabAgent(
            config=config,
            task_source=task_source_from_dir(tasks_root),
            system_prompt=prompts[SYSTEM_PROMPT_TARGET],
            task_template=prompts[TASK_TEMPLATE_TARGET],
            **agent_kwargs,
        )
        _assert_targets_applied(agent, prompts)
        output = await agent.forward(record=record)
        graded = await asyncio.to_thread(_grade, case=case, output=output, record=record, config=config)
    except HARNESS_ERRORS as exc:
        # The task tree, the sandbox, the model API, the judge API: none of
        # this is the agent's answer, so none of it may be scored as one.
        # Anything NOT listed in HARNESS_ERRORS is a bug and escapes to the
        # runtime, which records it as an unresolved case with a traceback.
        logger.exception("harvey-lab rollout failed: case_id=%s", case.case_id)
        # Retryable unless the condition is deterministic: re-running a stale
        # dataset row thirty times ends in the case being excluded, when what
        # the operator needs is to re-export it.
        retryable = not isinstance(exc, DatasetMismatchError)
        return CaseResult.failed(f"{type(exc).__name__}: {exc}", retryable=retryable)

    criteria = [dict(criterion) for criterion in case.ground_truth.get("criteria", ())]
    return CaseResult(
        output={
            ALL_PASS_FIELD: graded[ALL_PASS_FIELD],
            CRITERION_PASS_RATE_FIELD: graded[CRITERION_PASS_RATE_FIELD],
            "passed": graded["passed"],
            "total_criteria": graded["total_criteria"],
            "deliverables_produced": sorted(output.deliverables),
            "deliverables_missing": sorted(output.missing_deliverables),
            "finished": output.finished,
            "abandoned": output.abandoned,
            "max_turns_reached": output.max_turns_reached,
            "total_turns": output.total_turns,
            "final_answer": output.final_answer,
        },
        run_metrics={
            "trace_evidence": _trace_evidence(case=case, output=output, graded=graded, criteria=criteria),
            "timing": {"agent_seconds": round(output.wall_seconds, 3)},
        },
        context={
            # The verdicts the scorer aggregates, plus enough of the work product
            # to make a verdict legible during reflection.
            "verdicts": list(graded.get("verdicts", ())),
            "deliverable_previews": {
                name: text[:_PREVIEW_CHARS] for name, text in sorted(output.deliverables.items())
            },
        },
    )


# ── Scoring ──────────────────────────────────────────────────────────────────


class RubricScorer:
    """Aggregate the judge's per-criterion verdicts into the LAB metrics.

    Deterministic by design: the semantic judgment happened in ``run_case``, so
    this only counts verdicts. A case that never produced verdicts (a failed
    rollout) scores 0 on both fields — a real failure must deflate the metrics,
    never be excluded from them.
    """

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        criteria = [dict(criterion) for criterion in case.ground_truth.get("criteria", ())]
        total = len(criteria)
        verdicts = result.context.get("verdicts") or ()
        passed_ids = {str(verdict.get("id")) for verdict in verdicts if verdict.get("passed")}
        passed = sum(1 for criterion in criteria if str(criterion.get("id")) in passed_ids)
        rate = (passed / total) if total else 0.0
        scores = {
            ALL_PASS_FIELD: 1.0 if total and passed == total else 0.0,
            CRITERION_PASS_RATE_FIELD: rate,
        }
        return CaseScore(
            field_scores=scores,
            objective=objective_score(scores, field_weights=OBJECTIVE_WEIGHTS),
            key="harvey-lab-rubric",
        )


# ── Spec ─────────────────────────────────────────────────────────────────────


# Binaries the agent shells out to via ``code_exec``. They are named directly
# in the task prompt, and unlike the Python dependencies they cannot come from
# ``pyproject.toml`` — they are hand-listed in ``spec.apt_install`` in
# ``.beaker/beaker.yaml``, so that list can drift from what the agent
# actually needs. Missing binaries degrade the agent (it can read the task
# documents but not convert them) rather than crash it, so this warns once at
# spec load instead of failing every rollout in parallel.
_EXPECTED_AGENT_BINARIES = ("pandoc", "pdftotext", "soffice")


def _warn_on_missing_agent_binaries() -> None:
    """Log the document-conversion binaries the agent's shell is missing."""
    absent = [name for name in _EXPECTED_AGENT_BINARIES if shutil.which(name) is None]
    if absent:
        logger.warning(
            "Harvey LAB agent binaries not on PATH: %s. The agent can read task documents "
            "but not convert them; add them to `spec.apt_install` in .beaker/beaker.yaml.",
            ", ".join(absent),
        )


def _assert_judge_credential_present() -> None:
    """Fail at spec load if a hosted run cannot pay for the rubric judge.

    Only the *agent's* calls go through the run's inference gateway: it
    authorizes the models the customer selected for the run, and the judge is
    deliberately not one of them — it is a fixed grader, and letting it drift
    with the model under test would make two targets' scores incomparable. So
    the judge keeps calling its own provider directly and needs that provider's
    key in the run environment.

    Without it, every rollout would run the agent to completion and only then
    fail to grade it — the most expensive possible way to discover a missing
    environment variable.

    Temporary (contortion 3 in the module docstring). The key this asks for is
    the *same* OpenRouter credential the organization has already stored in
    Beaker settings, pasted a second time into the agent's sandbox secrets
    because only the gateway can read the first copy. It goes away when the
    gateway authorizes a run-scoped grader model alongside the model targets —
    which also removes the last provider credential from the sandbox, and lets
    grader spend be accounted separately instead of vanishing (contortion 4).
    """
    if not os.environ.get("BEAKER_INFERENCE_BASE_URL"):
        return  # Not a hosted run; the local CLI's own env rules apply.
    judge_model = HarveyLabConfig().judge_model
    provider = judge_model.split("/", 1)[0]
    key_variable = f"{provider.upper().replace('-', '_')}_API_KEY"
    if not os.environ.get(key_variable):
        raise RuntimeError(
            f"The rubric judge ({judge_model}) has no credential: {key_variable} is not set "
            "in this run's environment. The run's inference gateway covers the agent's model "
            "only — it authorizes the models selected for the run, and the judge is a fixed "
            f"grader outside that set. Set {key_variable} as a run environment variable."
        )


@spec(dataset_schema=DATASET_SCHEMA)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Assemble the Harvey LAB prompt-optimization spec."""
    del ctx
    _warn_on_missing_agent_binaries()
    _assert_judge_credential_present()
    return Spec(
        seed_targets=_seed_targets(),
        data_loader=LabTaskDataLoader(),
        run_case=_run_case,
        scorer=RubricScorer(),
    )
