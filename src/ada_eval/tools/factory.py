from enum import StrEnum
from pathlib import Path

from ada_eval.utils import construct_enum_case_insensitive

from .shell_script import ShellScript


class Tool(StrEnum):
    SHELL_SCRIPT = ShellScript.name

    # Constructor should be case-insensitive
    @classmethod
    def _missing_(cls, value):
        return construct_enum_case_insensitive(cls, value)


class UnsupportedToolError(Exception):
    def __init__(self, tool):
        super().__init__(f"Unsupported tool: {tool}")


def create_tool(tool: Tool, config_file: Path) -> ShellScript:
    match tool:
        case Tool.SHELL_SCRIPT:
            return ShellScript.from_config_file(config_file)
        case _:
            raise UnsupportedToolError(tool)
