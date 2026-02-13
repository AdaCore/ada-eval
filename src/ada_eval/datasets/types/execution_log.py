"""Execution log types for tracking agent actions."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class LogEntryType(StrEnum):
    """Type of execution log entry."""

    REASONING = "reasoning"
    TOOL_CALL = "tool_call"


class ExecutionLogEntry(BaseModel):
    """A single entry in the execution log."""

    type: LogEntryType
    timestamp_ms: int


class ReasoningLogEntry(ExecutionLogEntry):
    """A reasoning log entry, representing a piece of reasoning text."""

    type: Literal[LogEntryType.REASONING] = LogEntryType.REASONING
    text: str


class ToolCallLogEntry(ExecutionLogEntry):
    """A tool call log entry, representing a call to a tool."""

    type: Literal[LogEntryType.TOOL_CALL] = LogEntryType.TOOL_CALL
    tool_name: str
    tool_input: dict[str, Any] | None = None
    success: bool
    duration_ms: int


AnyLogEntry = ReasoningLogEntry | ToolCallLogEntry
