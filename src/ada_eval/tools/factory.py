from enum import Enum
from pathlib import Path

from ada_eval.tools.shell_script import ShellScript

from .generic_tool import GenericTool
from .spark_assistant import SparkAssistant


class Tool(Enum):
    SPARK_ASSISTANT = "spark_assistant"
    SHELL_SCRIPT = "shell_script"

    def __str__(self):
        return self.value


def create_tool(tool: Tool, config_file: Path) -> GenericTool:
    match tool:
        case Tool.SHELL_SCRIPT:
            return ShellScript.from_config_file(config_file)
        case Tool.SPARK_ASSISTANT:
            return SparkAssistant.from_config_file(config_file)
        case _:
            raise ValueError(f"Unsupported tool: {tool}")
