"""The Harvey LAB legal agent, driven through the Stirrup harness.

Stirrup (Artificial Analysis' agent framework) supplies the tool-use loop,
context management, and message plumbing. This module wires it up the way
AA's Harvey LAB-AA leaderboard does:

* **One tool.** The agent gets a single ``code_exec`` tool over a code
  execution environment — no curated ``read_document`` / ``write_deliverable``
  helpers. It must list, parse, and produce files itself (``pandoc``,
  ``python-docx``, ``openpyxl``, ``python-pptx``, …), which is what makes real
  ``.docx`` / ``.xlsx`` / ``.pptx`` deliverables possible and what LAB-AA means
  by "raw model ability".
* **AA's finish contract.** ``finish`` takes a summary plus the absolute paths
  of every deliverable, and *validates* that each path is a real file;
  ``abandon_task_finish`` lets the agent give up on a genuinely impossible
  task. Nothing outside a successful ``finish`` is graded.
* **Vision.** Stirrup's ``view_image`` tool is attached when enabled, reading
  images out of the environment as native image tokens.

``code_exec`` runs in a temp directory on this machine (Stirrup's local
backend) — **no isolation**; see the README. Both the execution environment and
the LLM client are injected through factories, so a sandboxed backend can be
dropped in without touching this module, and tests use a scripted client that
never hits the network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, cast

from pydantic import BaseModel, Field

from ..config import HarveyLabConfig
from ..data.dataset import HarveyLabRecord
from .prompts import load_harvey_lab_prompts
from .workspace import TaskSource, TaskWorkspace


logger = logging.getLogger(__name__)

FINISH_TOOL_NAME = "finish"
ABANDON_TOOL_NAME = "abandon_task_finish"


@dataclass
class HarveyLabAgentOutput:
    """Per-case result returned by the Harvey LAB Stirrup agent.

    ``deliverables`` maps each submitted, requested filename to its extracted
    text for the rubric judge; ``raw_deliverables`` holds the original bytes for
    ``run`` mode. Filenames are matched exactly, so ``missing_deliverables``
    lists what the task asked for and never received through ``finish``.
    ``submitted_paths`` records that finish call and ``abandoned`` records an
    ``abandon_task_finish`` give-up. Scoring happens in the evaluator.
    """

    final_answer: str
    deliverables: dict[str, str] = field(default_factory=dict)
    raw_deliverables: dict[str, bytes] = field(default_factory=dict)
    missing_deliverables: list[str] = field(default_factory=list)
    submitted_paths: list[str] = field(default_factory=list)
    finished: bool = False
    abandoned: bool = False
    max_turns_reached: bool = False
    total_turns: int = 0
    wall_seconds: float = 0.0


# Factory that builds a Stirrup ``LLMClient`` from model settings. Kept behind
# a factory so tests inject a scripted client and avoid network calls.
ModelFactory = Callable[[str, float, int, int, float, str], Any]

# Factory that builds the Stirrup ``CodeExecToolProvider`` used for each task.
# Kept injectable for tests and callers extending the local-only default.
ExecProviderFactory = Callable[[HarveyLabConfig], Any]


def _default_model_factory(
    model: str,
    temperature: float,
    max_tokens: int,
    context_window_tokens: int,
    timeout: float,
    reasoning_effort: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Build Stirrup's LiteLLM client (imported lazily; needs ``stirrup[litellm]``).

    ``api_base``/``api_key`` override where the call goes. Unset (the default),
    LiteLLM resolves the provider from the model slug and reads that provider's
    key from the environment. Set, the call is sent to one OpenAI-compatible
    endpoint instead — which is how a hosted Beaker rollout reaches its
    selected model through the run's inference gateway, holding no provider
    credential of its own.
    """
    from stirrup.clients.litellm_client import LiteLLMClient, ReasoningEffort

    # ``none``/empty is the documented "non-reasoning model" sentinel: send no
    # reasoning param at all.
    effort = reasoning_effort if reasoning_effort not in ("", "none") else None
    kwargs: dict[str, Any] = {"temperature": temperature, "timeout": timeout}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if effort is not None:
        # litellm only forwards ``reasoning_effort`` for models it already knows
        # are reasoning-capable; a newly released model (e.g. deepseek-v4-pro on
        # OpenRouter) isn't in that map yet, so litellm raises
        # ``UnsupportedParamsError`` even though OpenRouter accepts it. Opting in
        # via ``allowed_openai_params`` forwards the param as-is to the provider.
        kwargs["allowed_openai_params"] = ["reasoning_effort"]
    return LiteLLMClient(
        model=model,
        max_tokens=max_tokens,
        context_window_tokens=context_window_tokens,
        reasoning_effort=cast("ReasoningEffort | None", effort),
        api_key=api_key,
        kwargs=kwargs,
    )


def _default_exec_provider_factory(config: HarveyLabConfig) -> Any:
    """Build the local (temp-directory) code-execution environment for one task.

    A fresh provider is built per task: each owns its own temp directory, which
    is where the task's documents are staged and the deliverables are produced.

    **No isolation.** Stirrup's local backend runs the model's shell commands as
    the current user. That is a deliberate simplification — see the README.
    To harden it, pass your own ``exec_provider_factory`` to ``HarveyLabAgent``
    returning any other Stirrup ``CodeExecToolProvider``; the agent only uses
    the provider's generic file surface. The one backend-specific touchpoint is
    :func:`_env_working_dir` (the directory ``finish`` paths are built from),
    which reads the local backend's ``temp_dir`` — see its docstring.
    """
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    return LocalCodeExecToolProvider(shell_timeout=config.shell_timeout_s)


def _effective_deliverable_names(record: HarveyLabRecord) -> tuple[str, ...]:
    """Filenames the agent must produce, with a default when the task names none.

    A handful of LAB tasks declare no deliverable; AA still expects a single
    freeform ``response.md``. The task prompt, the deliverables collected for
    grading, and the ``missing_deliverables`` list all derive from this one
    helper, so a submitted ``response.md`` is graded rather than silently
    dropped (and correctly reported missing when absent).
    """
    return record.deliverable_names or ("response.md",)


def _render_template(template: str, variables: dict[str, str]) -> str:
    """Substitute ``{{name}}`` (or ``{{ name }}``) Jinja2-style variables.

    A variable the template never mentions is dropped rather than appended:
    unlike the old two-variable template, these prompts carry structural
    context (paths, tool names) that is meaningless out of position. The task
    ``instructions`` are the exception — the agent must always receive them, so
    a template missing them still gets them appended.
    """
    rendered = template
    for name, value in variables.items():
        placeholder = "{{%s}}" % name
        spaced = "{{ %s }}" % name
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, value)
        elif spaced in rendered:
            rendered = rendered.replace(spaced, value)
        elif name == "instructions":
            rendered = f"{rendered}\n\n{value}"
    return rendered


# ─── tool parameter models ────────────────────────────────────────────


class FinishParams(BaseModel):
    """AA's LAB-AA finish contract: a summary plus every deliverable path."""

    summary: str = Field(description="Brief summary of what you accomplished, including any assumptions you made.")
    paths: list[str] = Field(
        description="Absolute paths to every deliverable file you produced. Files only, not directories."
    )


class AbandonParams(BaseModel):
    """Give-up contract: a reason, no deliverables."""

    reason: str = Field(description="Why the task is genuinely impossible to complete.")


def _is_absolute_path(path: str) -> bool:
    """Whether ``path`` is absolute under POSIX *or* Windows rules.

    ``finish`` paths come from the execution environment, whose path style is
    the backend's, not the host's: a container/remote backend hands back POSIX
    paths (``/workspace/memo.md``) that host ``Path.is_absolute()`` would call
    relative on Windows. Accepting either style keeps the swap-in-a-backend
    extension working regardless of host OS.
    """
    return PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()


def _build_finish_tools(exec_env: Any) -> list[Any]:
    """Build LAB-AA's ``finish`` + ``abandon_task_finish`` pair.

    ``finish`` mirrors AA's validated submission: every submitted path must
    resolve to a real file in the execution environment, otherwise the call is
    rejected (``success=False``) and the agent gets another turn to fix it.
    Stirrup terminates the loop on whichever finish tool succeeds, and hands
    back its parameters — so the caller tells a completion from a give-up by
    the type of the returned params.
    """
    from stirrup import Tool, ToolResult

    async def _finish(params: FinishParams) -> Any:
        invalid: list[str] = []
        for path in params.paths:
            if not _is_absolute_path(path):
                invalid.append(path)
                continue
            try:
                exists = await exec_env.file_exists(path)
                is_dir = await exec_env.is_directory(path) if exists else False
            except Exception as exc:  # noqa: BLE001 - failed validation must reject finish
                logger.warning("finish could not stat %r: %s", path, exc)
                invalid.append(path)
                continue
            if not exists or is_dir:
                invalid.append(path)
        if invalid:
            return ToolResult(
                content=(
                    f"ERROR: these submitted paths are not absolute paths to existing files: {invalid}. "
                    "Verify the paths, save every deliverable as a real file under the exact "
                    "requested filename, then call finish again."
                ),
                success=False,
            )
        return ToolResult(content=params.summary)

    return [
        Tool(
            name=FINISH_TOOL_NAME,
            description=(
                "Submit your work: a brief summary plus the absolute paths of every deliverable "
                "file. Anything not submitted here is not graded."
            ),
            parameters=FinishParams,
            executor=_finish,
        ),
        Tool(
            name=ABANDON_TOOL_NAME,
            description=(
                "Give up on the task, with a brief reason. Use only when you have concluded the "
                "work is genuinely impossible — not to escape difficulty."
            ),
            parameters=AbandonParams,
            executor=lambda params: ToolResult(content=params.reason),
        ),
    ]


class HarveyLabAgent:
    """A Stirrup-driven legal agent over a single ``code_exec`` tool.

    ``forward`` stages the task's documents, spins up a code execution
    environment, runs the Stirrup loop under AA's LAB-AA prompts, and pulls the
    requested deliverables back out for the rubric judge.
    """

    def __init__(
        self,
        *,
        config: HarveyLabConfig,
        task_source: TaskSource,
        model_factory: ModelFactory | None = None,
        exec_provider_factory: ExecProviderFactory | None = None,
        system_prompt: str | None = None,
        task_template: str | None = None,
    ) -> None:
        self._config = config
        self._task_source = task_source
        self._model_factory = model_factory or _default_model_factory
        self._exec_provider_factory = exec_provider_factory or _default_exec_provider_factory
        default_system, default_task = load_harvey_lab_prompts()
        self._system_prompt = _render_template(
            system_prompt if system_prompt is not None else default_system,
            {
                "max_turns": str(config.max_turns),
                "finish_tool": FINISH_TOOL_NAME,
                "abandon_tool": ABANDON_TOOL_NAME,
            },
        )
        self._task_template = task_template if task_template is not None else default_task

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def task_template(self) -> str:
        return self._task_template

    def render_task_prompt(self, record: HarveyLabRecord, *, workspace_dir: str, documents_dir: str) -> str:
        """Render AA's task prompt for ``record`` against the live env paths.

        The working directory only exists once the execution environment is up,
        so the prompt is rendered inside the session rather than at construction.
        """
        names = _effective_deliverable_names(record)
        return _render_template(
            self._task_template,
            {
                "workspace_dir": workspace_dir,
                "documents_dir": documents_dir,
                "command_timeout_minutes": str(max(1, round(self._config.shell_timeout_s / 60))),
                "finish_tool": FINISH_TOOL_NAME,
                "abandon_tool": ABANDON_TOOL_NAME,
                "title": record.title or record.task_id,
                "instructions": record.instructions,
                "deliverables": "\n".join(f"- `{name}`" for name in names),
            },
        )

    async def forward(self, *, record: HarveyLabRecord) -> HarveyLabAgentOutput:
        from stirrup import Agent

        workspace = self._task_source(record)
        started = time.monotonic()
        try:
            exec_env = self._exec_provider_factory(self._config)
            tools: list[Any] = [exec_env]
            if self._config.enable_view_image:
                from stirrup.tools.view_image import ViewImageToolProvider

                tools.append(ViewImageToolProvider(exec_env))
            client = self._model_factory(
                self._config.task_model,
                self._config.task_temperature,
                self._config.max_output_tokens,
                self._config.context_window_tokens,
                self._config.task_llm_timeout,
                self._config.task_reasoning_effort,
            )
            agent: Any = Agent(
                client=client,
                name="harvey-lab",
                system_prompt=self._system_prompt,
                tools=tools,
                finish_tool=_build_finish_tools(exec_env),
                max_turns=self._config.max_turns,
            )
            # cache_on_interrupt=False: the eval may run cases in worker threads,
            # where Stirrup's default SIGINT handler raises "signal only works in
            # main thread of the main interpreter".
            async with agent.session(output_dir=workspace.output_dir, cache_on_interrupt=False) as session:
                # Stage documents explicitly at `documents/`; uploading the
                # workspace root would add an unwanted directory level.
                await _upload_documents(exec_env, workspace)
                work_dir = _env_working_dir(exec_env)
                user_prompt = self.render_task_prompt(
                    record,
                    workspace_dir=work_dir,
                    documents_dir="documents",
                )
                finish_params, history, _metadata = await session.run(user_prompt)
            abandoned = isinstance(finish_params, AbandonParams)
            submitted_paths = list(getattr(finish_params, "paths", []) or [])
            names = _effective_deliverable_names(record)
            deliverables = workspace.collect_deliverables(names)
            raw_deliverables = {name: workspace.deliverable_path(name).read_bytes() for name in deliverables}
            return HarveyLabAgentOutput(
                final_answer=_finish_message(finish_params),
                deliverables=deliverables,
                raw_deliverables=raw_deliverables,
                missing_deliverables=[n for n in names if n not in deliverables],
                submitted_paths=submitted_paths,
                finished=isinstance(finish_params, FinishParams),
                abandoned=abandoned,
                max_turns_reached=finish_params is None,
                total_turns=_count_turns(history),
                wall_seconds=time.monotonic() - started,
            )
        finally:
            workspace.close()


def _count_turns(history: list[list[Any]]) -> int:
    """Count accepted assistant turns across Stirrup's compacted history chunks."""
    return sum(1 for chunk in history for message in chunk if getattr(message, "role", None) == "assistant")


def _finish_message(finish_params: Any) -> str:
    """The agent's closing text: a ``finish`` summary or an abandon reason."""
    if finish_params is None:  # max_turns exhausted without any finish call
        return ""
    return str(getattr(finish_params, "summary", None) or getattr(finish_params, "reason", "") or "")


def _env_working_dir(exec_env: Any) -> str:
    """Absolute path the agent's shell runs in, used to build ``finish`` paths.

    Resolving this is the *one* backend-specific touchpoint: the local backend
    exposes it as ``temp_dir``; a container/remote backend runs commands in its
    own working directory (e.g. Stirrup's Docker backend uses ``/workspace``),
    so a swapped-in ``exec_provider_factory`` must also make that directory
    discoverable here. Prefers an explicit ``working_dir`` if a backend
    provides one, else falls back to ``temp_dir``.
    """
    value = getattr(exec_env, "working_dir", None) or getattr(exec_env, "temp_dir", None)
    if value is None:
        raise RuntimeError(
            f"{type(exec_env).__name__} exposes no working directory (no `working_dir`/`temp_dir`). "
            "If you injected a custom exec_provider_factory, teach _env_working_dir its working directory."
        )
    return str(value)


async def _upload_documents(exec_env: Any, workspace: TaskWorkspace) -> None:
    """Copy the task's staged documents into the local environment at ``documents/``."""
    result = await exec_env.upload_files(workspace.documents_dir, dest_dir="documents")
    failed = getattr(result, "failed", None)
    if failed:
        raise RuntimeError(f"Failed to stage task documents: {failed}")


__all__ = [
    "ExecProviderFactory",
    "HarveyLabAgent",
    "HarveyLabAgentOutput",
    "ModelFactory",
]
