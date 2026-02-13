"""Claude Agent tool using the Anthropic Agent SDK."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookMatcher,
    ResultMessage,
    TextBlock,
)
from pydantic import field_validator

from ada_eval.datasets import GeneratedSparkSample, SparkSample, get_contents
from ada_eval.datasets.types import ExitStatus, GenerationStats
from ada_eval.datasets.types.execution_log import (
    AnyLogEntry,
    ReasoningLogEntry,
    ToolCallLogEntry,
)

from .generic_tool import BaseConfig, GenericTool
from .tool_type import ToolType

logger = logging.getLogger(__name__)


class MCPConfigFileMissingError(FileNotFoundError):
    """Raised when the specified MCP config file is not found."""

    def __init__(self, mcp_config_path: Path):
        super().__init__(f"MCP config file not found at path: {mcp_config_path}")


class ClaudeAgentConfig(BaseConfig):
    """Configuration for the Claude Agent SDK tool."""

    tool: ToolType = ToolType.CLAUDE_AGENT

    # Model configuration
    model: str | None = None  # None = SDK default

    # Limits (no max_turns - use budget and timeout instead)
    max_budget_usd: float | None = None

    # MCP configuration - path to MCP config file (resolved relative to config file)
    mcp_config: Path | None = None

    # Tool restrictions - empty means use SDK defaults
    disallowed_tools: ClassVar[list[str]] = []

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, v: ToolType) -> ToolType:
        if v != ToolType.CLAUDE_AGENT:
            msg = f"Expected tool type 'claude_agent', got '{v}'"
            raise ValueError(msg)
        return v

    @classmethod
    def from_file(cls, config_file: Path) -> Self:
        """Load the configuration from a JSON file."""
        config = cls.model_validate_json(config_file.read_text(encoding="utf-8"))
        # Resolve the MCP config path relative to the config file's directory
        if config.mcp_config is not None:
            resolved_mcp_path = (config_file.parent / config.mcp_config).resolve()
            if not resolved_mcp_path.is_file():
                raise MCPConfigFileMissingError(resolved_mcp_path)
            return config.model_copy(update={"mcp_config": resolved_mcp_path})
        return config


def _get_usage_field(
    result_message: object,
    field: str,
) -> int | None:
    """Safely extract a usage field from the result message."""
    if not result_message:
        return None
    usage = getattr(result_message, "usage", None)
    if not usage:
        return None
    if isinstance(usage, dict):
        return usage.get(field)
    return getattr(usage, field, None)


async def _pre_tool_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    current_tool_start: dict[str, int],
    elapsed_ms: Callable[[], int],
) -> dict[str, Any]:
    """Record tool call start time."""
    if tool_use_id:
        current_tool_start[tool_use_id] = elapsed_ms()
    tool_name = input_data.get("tool_name", "unknown")
    logger.debug("Tool started: %s (id=%s)", tool_name, tool_use_id)
    return {}


async def _post_tool_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    current_tool_start: dict[str, int],
    execution_log: list[AnyLogEntry],
    elapsed_ms: Callable[[], int],
) -> dict[str, Any]:
    """Log tool call completion to execution log."""
    now_ms = elapsed_ms()
    start_ms = current_tool_start.pop(tool_use_id, now_ms) if tool_use_id else now_ms

    # Check if tool succeeded
    tool_response = input_data.get("tool_response")
    is_error = isinstance(tool_response, dict) and tool_response.get("is_error", False)

    # Capture tool input arguments (e.g. resource URIs for MCP tools)
    tool_input = input_data.get("tool_input")

    tool_name = input_data.get("tool_name", "unknown")
    execution_log.append(
        ToolCallLogEntry(
            timestamp_ms=start_ms,
            tool_name=tool_name,
            tool_input=tool_input if isinstance(tool_input, dict) else None,
            success=not is_error,
            duration_ms=now_ms - start_ms,
        )
    )
    return {}


async def _run_agent(
    options: ClaudeAgentOptions,
    prompt: str,
    sample_name: str,
    timeout_s: int,
    execution_log: list[AnyLogEntry],
    elapsed_ms: Callable[[], int],
) -> tuple[ExitStatus, "ResultMessage | None"]:
    """
    Run the Agent SDK client and collect results.

    Returns the exit status and the final ResultMessage (if any).
    """
    exit_status = ExitStatus.SUCCESS
    result_message: ResultMessage | None = None
    timed_out = False

    # Grace period after interrupt to receive final ResultMessage
    interrupt_grace_s = 60

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            async def _interrupt_after_timeout() -> None:
                """Wait for the configured timeout, then interrupt."""
                nonlocal timed_out
                await asyncio.sleep(timeout_s)
                timed_out = True
                logger.warning(
                    "Timeout reached (%ds) for sample %s, "
                    "sending interrupt to collect stats",
                    timeout_s,
                    sample_name,
                )
                await client.interrupt()

            timeout_task = asyncio.create_task(_interrupt_after_timeout())

            try:
                # Hard deadline: timeout + grace period, as a safety net
                # in case interrupt doesn't produce a ResultMessage.
                hard_deadline = timeout_s + interrupt_grace_s
                async with asyncio.timeout(hard_deadline):
                    async for message in client.receive_response():
                        # Capture reasoning from assistant messages
                        if isinstance(message, AssistantMessage):
                            execution_log.extend(
                                ReasoningLogEntry(
                                    timestamp_ms=elapsed_ms(), text=block.text
                                )
                                for block in message.content
                                if isinstance(block, TextBlock) and block.text
                            )

                        # Capture final result
                        if isinstance(message, ResultMessage):
                            result_message = message
                            if message.is_error:
                                exit_status = ExitStatus.SDK_ERROR
            except TimeoutError:
                # Hard deadline hit - interrupt didn't produce a result
                timed_out = True
                logger.warning(
                    "Hard timeout reached (%ds) for sample %s",
                    hard_deadline,
                    sample_name,
                )
            finally:
                timeout_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await timeout_task

    except Exception:
        exit_status = ExitStatus.SDK_ERROR
        logger.exception("Agent failed for sample %s", sample_name)

    if timed_out:
        exit_status = ExitStatus.TIMEOUT

    return exit_status, result_message


class ClaudeAgent(GenericTool[ClaudeAgentConfig, SparkSample, GeneratedSparkSample]):
    """
    Tool that uses the Anthropic Agent SDK to generate completions.

    Provides rich telemetry including cost tracking, token usage,
    and an interleaved execution log of reasoning and tool calls.

    Uses ClaudeSDKClient (not query()) to support hooks for logging.
    """

    name: ClassVar[Literal["claude_agent"]] = "claude_agent"
    type_map: ClassVar = {SparkSample: GeneratedSparkSample}
    config_type = ClaudeAgentConfig

    def apply(self, sample: SparkSample) -> GeneratedSparkSample:
        """Apply the agent to a sample (sync wrapper for async implementation)."""
        return asyncio.run(self._apply_async(sample))

    async def _apply_async(self, sample: SparkSample) -> GeneratedSparkSample:
        """Run the Agent SDK and collect results."""

        with sample.sources.unpacked() as sample_working_dir:
            logger.debug(
                "Applying ClaudeAgent to %s in %s", sample.name, sample_working_dir
            )

            # Tracking state - single in-order log
            execution_log: list[AnyLogEntry] = []
            start_time = time.monotonic()
            current_tool_start: dict[str, int] = {}  # tool_use_id -> start_ms
            stderr_buffer: list[str] = []

            def elapsed_ms() -> int:
                return int((time.monotonic() - start_time) * 1000)

            # Hook callbacks
            async def pre_tool_hook(
                input_data: dict[str, Any],
                tool_use_id: str | None,
                _context: HookContext,
            ) -> dict[str, Any]:
                """Record tool call start time."""
                return await _pre_tool_hook(
                    input_data, tool_use_id, current_tool_start, elapsed_ms
                )

            async def post_tool_hook(
                input_data: dict[str, Any],
                tool_use_id: str | None,
                _context: HookContext,
            ) -> dict[str, Any]:
                """Log tool call completion to execution log."""
                return await _post_tool_hook(
                    input_data,
                    tool_use_id,
                    current_tool_start,
                    execution_log,
                    elapsed_ms,
                )

            # Build options - use SDK defaults for allowed_tools
            disallowed = (
                list(self.config.disallowed_tools)
                if self.config.disallowed_tools
                else []
            )
            # We need to set the permission mode to bypassPermissions to allow
            # the agent to use tools without prompting for permissions, which
            # would be undesirable in an automated evaluation setting
            options = ClaudeAgentOptions(
                cwd=str(sample_working_dir),
                permission_mode="bypassPermissions",
                max_budget_usd=self.config.max_budget_usd,
                model=self.config.model,
                disallowed_tools=disallowed,
                mcp_servers=self.config.mcp_config or {},
                stderr=stderr_buffer.append,
                hooks={
                    "PreToolUse": [HookMatcher(hooks=[pre_tool_hook])],  # type: ignore[list-item]
                    "PostToolUse": [HookMatcher(hooks=[post_tool_hook])],  # type: ignore[list-item]
                },
            )

            exit_status, result_message = await _run_agent(
                options=options,
                prompt=sample.prompt,
                sample_name=sample.name,
                timeout_s=self.config.timeout_s,
                execution_log=execution_log,
                elapsed_ms=elapsed_ms,
            )

            # Calculate runtime
            runtime_ms = elapsed_ms()

            # Check for budget exceeded
            if (
                result_message
                and result_message.total_cost_usd
                and self.config.max_budget_usd
                and result_message.total_cost_usd >= self.config.max_budget_usd
            ):
                exit_status = ExitStatus.BUDGET_EXCEEDED

            # Build stats
            generation_stats = GenerationStats(
                exit_status=exit_status,
                stdout="",
                stderr="".join(stderr_buffer),
                runtime_ms=runtime_ms,
                total_cost_usd=(
                    getattr(result_message, "total_cost_usd", None)
                    if result_message
                    else None
                ),
                input_tokens=_get_usage_field(result_message, "input_tokens"),
                output_tokens=_get_usage_field(result_message, "output_tokens"),
                cache_read_tokens=_get_usage_field(
                    result_message, "cache_read_input_tokens"
                ),
                cache_creation_tokens=_get_usage_field(
                    result_message, "cache_creation_input_tokens"
                ),
                num_turns=(
                    getattr(result_message, "num_turns", None)
                    if result_message
                    else None
                ),
                session_id=(
                    getattr(result_message, "session_id", None)
                    if result_message
                    else None
                ),
                model=self.config.model,
                execution_log=execution_log,
            )

            # Pack results
            generated_files = get_contents(sample_working_dir)

            return GeneratedSparkSample(
                **sample.model_dump(),
                generated_solution=generated_files,
                generation_stats=generation_stats,
            )
