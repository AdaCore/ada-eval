from pathlib import Path

from .generic_tool import GenericTool
from .spark_assistant import SparkAssistant


def create_tool(tool_name: str, config_file: Path) -> GenericTool:
    match tool_name:
        case "spark_assistant":
            return SparkAssistant.from_config_file(config_file)
        case _:
            raise ValueError(f"Unsupported tool: {tool_name}")
