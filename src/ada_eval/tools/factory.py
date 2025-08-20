from enum import Enum
from pathlib import Path

from .shell_script import ShellScript


class Tool(Enum):
    SHELL_SCRIPT = ShellScript.name

    def __str__(self):
        return self.value


class UnsupportedToolError(Exception):
    def __init__(self, tool):
        super().__init__(f"Unsupported tool: {tool}")


def create_tool(tool: Tool, config_file: Path) -> ShellScript:
    match tool:
        case Tool.SHELL_SCRIPT:
            return ShellScript.from_config_file(config_file)
        case _:
            raise UnsupportedToolError(tool)
