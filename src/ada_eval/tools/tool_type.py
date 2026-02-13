from enum import StrEnum

from ada_eval.utils import construct_enum_case_insensitive


class ToolType(StrEnum):
    """Type of tool used for generation."""

    SHELL_SCRIPT = "shell_script"
    CLAUDE_AGENT = "claude_agent"

    # Makes the constructor case-insensitive
    @classmethod
    def _missing_(cls, value):
        return construct_enum_case_insensitive(cls, value)
