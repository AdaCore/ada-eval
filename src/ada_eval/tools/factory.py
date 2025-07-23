from enum import Enum
from pathlib import Path

from .generic_tool import GenericTool
from .spark_assistant import SparkAssistant


class Tool(Enum):
    SPARK_ASSISTANT = "spark_assistant"

    def __str__(self):
        return self.value


def create_tool(tool: Tool, config_file: Path) -> GenericTool:
    match tool:
        case Tool.SPARK_ASSISTANT:
            return SparkAssistant.from_config_file(config_file)
        case _:
            raise ValueError(f"Unsupported tool: {tool}")
