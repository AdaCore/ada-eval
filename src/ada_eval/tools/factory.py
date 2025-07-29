from enum import Enum
from pathlib import Path

from ada_eval.tools.shell_script import ShellScript

from .generic_tool import GenerationTool


class GenerationToolName(Enum):
    SHELL_SCRIPT = "shell_script"

    def __str__(self):
        return self.value


class UnsupportedToolError(Exception):
    def __init__(self, tool):
        super().__init__(f"Unsupported tool: {tool}")


def create_generation_tool(
    tool: GenerationToolName, config_file: Path
) -> GenerationTool:
    match tool:
        case GenerationToolName.SHELL_SCRIPT:
            return ShellScript.from_config_file(config_file)
        case _:
            raise UnsupportedToolError(tool)
